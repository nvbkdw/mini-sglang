from functools import cached_property
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig
import torch
from typing import Dict, List, Tuple
from uuid import uuid4
from dataclasses import dataclass

from transformers.tokenization_utils_base import BatchEncoding
from urllib3 import request
from nano.models.qwen3 import Qwen3ForCausalLM
from nano.models.config import ModelConfig
from nano.models.config import _load_config
from nano.utils import load_hf_weight
from nano.sampler import Sampler
from nano.core import Batch, Req, SamplingParams
from nano.kvcache.cache import KVCache

model_name = "Qwen/Qwen3-0.6B"

def mem_GB(size: int) -> str:
    return f"{size / (1024**3):.2f} GiB"

@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    dtype: torch.dtype = torch.bfloat16
    max_running_req: int = 256
    attention_backend: str = "auto"
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages

    @cached_property
    def hf_config(self):
        return _load_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        from minisgl.models import ModelConfig

        return ModelConfig.from_hf(self.hf_config)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:23333"
        
        
class Engine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.model_name = config.model_path
        # TODO: pick GPU based on TP rank
        self.device = torch.device("cuda:0")
        self.dtype = config.dtype
        torch.set_default_device(self.device)
        
        self.model_hf_config =  config.hf_config
        self.model_config = ModelConfig.from_hf(self.model_hf_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = Qwen3ForCausalLM(self.model_config)
        self.sampler = Sampler(self.device)
        self.request_uid_counter = 0
        
        # initialize the model
        self.model.load_state_dict(self._load_weight_state_dict(), strict=False)
        self.model.to(self.device, self.dtype)
        free_memory = self._sync_get_memory()
        print(f"Free memory after loading model: {mem_GB(free_memory)}")
        available_memory = int(config.memory_ratio * free_memory)
        self._allocate_kv_cache(available_memory)
        
    
    def _sync_get_memory(self) -> Tuple[int, int]:
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = torch.cuda.mem_get_info(self.device)[0]
        # TODO: get min and max free memory across TP ranks
        return free_memory
    
    def _allocate_kv_cache(self, available_memory: int) -> None:
        # allocate page table for KV cache
        cache_per_page = (
            2  
            * self.model_config.head_dim # head dimension
            * self.model_config.num_kv_heads # number of key-value heads
            * self.config.page_size # number of tokens per page
            * self.config.dtype.itemsize
            * self.model_config.num_layers
        )
        num_pages = available_memory // cache_per_page
        
        self.kv_cache = KVCache(
            num_pages=num_pages,
            num_kv_heads=self.model_config.num_kv_heads,
            num_layers=self.model_config.num_layers,
            head_dim=self.model_config.head_dim,
            dtype=self.config.dtype,
            device=self.device,
        ) 


    def _load_weight_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            k: v.to(self.dtype)
            for k, v in load_hf_weight(self.model_name, self.device).items()
        }
        
    def generate_batch(self, prompts: List[str], max_new_tokens: int = 2000) -> torch.Tensor:
        # prepare the model input batch
        input_batch = self._prepare_batch(prompts)
        # TODO: store output ids in page table
        output_ids = torch.empty((input_batch.size, max_new_tokens), dtype=torch.int32, device=self.device) # [batch, max_new_tokens]

        for i in range(max_new_tokens):
            next_tokens = self.forward(input_batch)  # [batch, 1] on GPU
            output_ids[:,i] = next_tokens[:]
            # end of sequence
            # TODO: allow batch to have different lengths
            if (next_tokens[:] == self.tokenizer.eos_token_id).all():
                print(f"End of sequence at token {i}")
                break
            # Append next tokens to each request's host_ids
            for j, req in enumerate(input_batch.reqs):
                req.append_host(next_tokens[j])

        return self._postprocess_output(output_ids[:,:i+1])

    def _prepare_batch(self, prompts: List[str]) -> Batch:
        # prepare the model input
        requests: List[Req] = []
        for prompt in prompts:
            messages = [
                {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
            )
            # TODO: don't put all token to device, use page table to manage the token ids
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            req = Req(
                input_ids=model_inputs['input_ids'], 
                output_len=2048, 
                uid=self.request_uid_counter, 
                sampling_params=SamplingParams())
            self.request_uid_counter += 1
            requests.append(req)
        batch = Batch(reqs=requests, phase="decode")
        return batch

    def _postprocess_output(self, output_ids: torch.Tensor) -> str:
        content = [self.tokenizer.decode(output_ids[i], skip_special_tokens=False).strip("\n") for i in range(output_ids.shape[0])]
        return content
    
    def forward(self, batch: Batch) -> torch.Tensor:
        # conduct text completion
        # TODO: use page table to manage the model inputs
        # Stack requests: each host_ids is [seq_len], stack to get [batch, seq_len]
        model_inputs = torch.stack([req.host_ids for req in batch.reqs], dim=0)
        model_inputs = model_inputs.to(self.device).contiguous()
        logits = self.model(model_inputs)
        # Only use the last position's logits for next token prediction
        last_logits = logits[:, -1, :]  # [batch, vocab_size]
        # Sample next token
        next_tokens = self.sampler.sample(last_logits).to(torch.int32)  # [batch, 1]
        return next_tokens