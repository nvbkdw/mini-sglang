from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig
import torch
from typing import Dict, List
from uuid import uuid4
from transformers.tokenization_utils_base import BatchEncoding
from urllib3 import request
from nano.models.qwen3 import Qwen3ForCausalLM
from nano.models.config import ModelConfig
from nano.models.config import _load_config
from nano.utils import load_hf_weight
from nano.sampler import Sampler
from nano.core import Batch, Req, SamplingParams

model_name = "Qwen/Qwen3-0.6B"

class Engine:
    def __init__(self, model_name: str):
        self.model_name = model_name
        # TODO: pick GPU based on TP rank
        self.device = torch.device("cuda:0")
        self.dtype = torch.bfloat16
        torch.set_default_device(self.device)
        self.model_hf_config =  _load_config(model_name)
        self.model_config = ModelConfig.from_hf(self.model_hf_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = Qwen3ForCausalLM(self.model_config)
        self.sampler = Sampler(self.device)
        self.request_uid_counter = 0
        
        
        # initialize the model
        self.model.load_state_dict(self._load_weight_state_dict(), strict=False)
        self.model.to(self.device, self.dtype)
        
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
            output_ids[:,i] = next_tokens[:,0]
            # end of sequence
            # TODO: allow batch to have different lengths
            if (next_tokens[:,0] == self.tokenizer.eos_token_id).all():
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
        content = self.tokenizer.decode(output_ids, skip_special_tokens=False).strip("\n")
        return content
    
    def forward(self, batch: Batch) -> torch.Tensor:
        # conduct text completion
        # TODO: use page table to manage the model inputs
        # Stack requests: each host_ids is [seq_len], stack to get [batch, seq_len]
        model_inputs = torch.stack([req.host_ids for req in batch.reqs], dim=0)
        model_inputs = model_inputs.to(self.device).contiguous()
        breakpoint()
        logits = self.model(model_inputs)
        # Only use the last position's logits for next token prediction
        last_logits = logits[:, -1, :]  # [batch, vocab_size]
        # Sample next token
        next_tokens = self.sampler.sample(last_logits).to(torch.int32)  # [batch, 1]
        return next_tokens