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

from nano.models.ops.attention import HybridBackend, FlashInferBackend

model_name = "Qwen/Qwen3-0.6B"

def mem_GB(size: int) -> str:
    return f"{size / (1024**3):.2f} GiB"

def _make_2d_indices(table_2d: torch.Tensor, ranges: List[Tuple[int, int, int]]) -> torch.Tensor:
    """
    Return the 1D indices for the given 2D table and ranges.

    Example: The underlying indices of a 2D table (3, 4) are:
        [[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]]
    For ranges [(0, 1, 3), (2, 0, 2)], the returned indices are [1, 2, 8, 9].

    Args:
        table_2d (torch.Tensor): The 2D table tensor.
        ranges (List[Tuple[int, int, int]]): A list of tuples (entry, begin, end),
            where `entry` is the row index in the 2D table, and `begin` and `end`
            specify the range of column indices to include.
    Returns:
        torch.Tensor: A 1D tensor of indices.
    """
    assert table_2d.dim() == 2 and table_2d.is_contiguous()
    STRIDE = table_2d.stride(0)
    needed_size = sum(end - begin for _, begin, end in ranges)
    indices_host = torch.empty(needed_size, dtype=torch.int32)
    offset = 0
    for entry, begin, end in ranges:
        length = end - begin
        offset += length
        torch.arange(
            begin + entry * STRIDE,
            end + entry * STRIDE,
            dtype=torch.int32,
            out=indices_host[offset - length : offset],
        )
    return indices_host.to(table_2d.device, non_blocking=True)

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

class CacheManager:
    
    def __init__(self, page_table: torch.Tensor, kv_cache: KVCache):
        self.page_table = page_table
        self.kv_cache = kv_cache
        
        # page table rows
        self.free_pt_rows = list(range(self.page_table.shape[0]))
        self.req_to_row = {}
        
        # free page indices
        self.free_page_indices = list(range(self.kv_cache.num_pages))
        
        
    def allocate_pages(self, req: Req) -> None:
        """ Allocate pages for the request.
        """
        
        # allocate a row in virtual page table
        row_idx = self.free_pt_rows.pop(0)
        self.req_to_row[req.uid] = row_idx
        req.table_idx = row_idx
        
        # allocate pages in physical KV cache
        for i in range(req.device_len):
            page_idx = self.free_page_indices.pop(0)
            self.page_table[row_idx, i] = page_idx
            
        return row_idx
        
        
    def free_pages(self, req: Req) -> None:
        """ Free pages for the request.
        """
        
        # free the row for the request
        row_idx = self.req_to_row[req.uid]
        self.free_pt_rows.append(row_idx)
        del self.req_to_row[req.uid]
        
        # release the pages for the request
        for i in range(req.device_len):
            page_idx = self.page_table[row_idx, i]
            self.free_page_indices.append(page_idx)
            
    def allocate(self, needed_size: int) -> torch.Tensor:
        """ Allocate space before batch forward.
        """
        if needed_size > (free_len := len(self.free_page_indices)):
            raise ValueError(f"Not enough free pages to allocate {needed_size} pages.")
        allocated = self.free_page_indices[:needed_size]
        self.free_page_indices = self.free_page_indices[needed_size:]
        return allocated
        
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
        self.max_new_tokens = 2048
        
        # initialize the model
        self.model.load_state_dict(self._load_weight_state_dict(), strict=False)
        self.model.to(self.device, self.dtype)
        free_memory = self._sync_get_memory()
        print(f"Free memory after loading model: {mem_GB(free_memory)}")
        available_memory = int(config.memory_ratio * free_memory)
        
        # create physical KV cache
        self.kv_cache = self._create_kv_cache(available_memory)
        
        # create virtual page table 
        self.page_table = torch.zeros(
            (config.max_running_req, self.max_new_tokens), 
            dtype=torch.int32, 
            device=self.device,
        )

        self.cache_manager = CacheManager(self.page_table, self.kv_cache)
        
    
    def _sync_get_memory(self) -> Tuple[int, int]:
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = torch.cuda.mem_get_info(self.device)[0]
        # TODO: get min and max free memory across TP ranks
        return free_memory
    
    def _create_kv_cache(self, available_memory: int) -> None:
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
        
        return KVCache(
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
        
    def generate_batch(self, prompts: List[str]) -> torch.Tensor:
        # prepare the model input batch
        input_batch = self._create_prefill_batch(prompts)

        for i in range(self.max_new_tokens):
            next_tokens = self.forward(input_batch)  # [batch, 1] on GPU
            
            # complete one step for each request
            for j, req in enumerate(input_batch.reqs):
                req.complete_one(next_tokens[j])
            
            # prepare next batch for decode
            input_batch = self._prepare_batch(input_batch)
                
            # end of sequence
            # TODO: allow batch to have different lengths
            if (next_tokens[:] == self.tokenizer.eos_token_id).all():
                print(f"End of sequence at token {i}")
                break
            
            # Append next tokens to each request's host_ids
            for j, req in enumerate(input_batch.reqs):
                req.append_host(next_tokens[j])

        # free resources for finished request batch
        for req in input_batch.reqs:
            self.cache_manager.free_pages(req)

        return self._postprocess_output(input_batch)

    def _create_prefill_batch(self, prompts: List[str]) -> Batch:
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
            
            # allocate page table and kvcache pages for the request
            # TODO: integrate with RadixTree
            self.cache_manager.allocate_pages(req)
        
        batch = Batch(reqs=requests, phase="prefill")
        
        # initialize the attention backend
        attn_backend = HybridBackend(
            prefill_backend=FlashInferBackend(
                config=self.model_config,
                kvcache=self.kv_cache,
                page_table=self.page_table,
                device=self.device,
                dtype=self.dtype,
            ),
            decode_backend=FlashInferBackend(
                config=self.model_config,
                kvcache=self.kv_cache,
                page_table=self.page_table,
                device=self.device,
                dtype=self.dtype,
            ),
        )
        
        batch.attn_backend = attn_backend
        
        return self._prepare_batch(batch)
    
    def _prepare_batch(self, batch: Batch) -> Batch:
        """ Prepare batch read and write indices of page table.
        """
        
        needed_size = sum(r.extend_len for r in batch.reqs)
        # allocate new pages for the batch
        out_loc = self.cache_manager.allocate(needed_size)
        
        # compute indices for loading token IDs from page table
        batch.load_indices = _make_2d_indices(
            self.page_table,
            [(r.table_idx, r.cached_len, r.device_len) for r in batch.reqs]
        )
        batch.write_indices = _make_2d_indices(
            self.page_table,
            [(r.table_idx, r.device_len, r.device_len + 1) for r in batch.reqs]
        )
        self.page_table.view(-1)[batch.write_indices] = out_loc
        
        # prepare attention metadata
        batch.attn_backend.prepare_metadata(batch)
        
        return batch 

    def _postprocess_output(self, batch: Batch) -> str:
        content = [self.tokenizer.decode(req.host_ids, skip_special_tokens=False).strip("\n") for req in batch.reqs]
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