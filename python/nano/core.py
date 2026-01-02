from dataclasses import dataclass
import torch
from abc import ABC, abstractmethod
from typing import List, Literal

@dataclass
class SamplingParams:
    top_k: int = 1
    ignore_eos: bool = False
    temperature: float = 0.0
    max_tokens: int = 2048
    
    
@dataclass
class Req:
    def __init__(self, 
        *,
        input_ids: torch.Tensor,
        output_len: int,
        uid: int,
        sampling_params: SamplingParams,
    ) -> None:
        self.uid = uid
        self.sampling_params = sampling_params

        # token genearation metadata
        # assert input_ids.is_cpu
        # Ensure input_ids is 1D [seq_len] for storage
        self.host_ids = input_ids.squeeze(0) if input_ids.dim() > 1 else input_ids
        # cached tokens
        self._cached_len = 0
        # total len to generate
        self.device_len = self.host_ids.shape[0]
        # max len to generate
        self.max_device_len = len(input_ids) + output_len
        assert 0 < self.device_len <= self.max_device_len
        
        #  row index in page table
        self.table_idx = -1
        
    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len
    
    @property
    def extend_len(self) -> int:
        return self.device_len - self._cached_len
    
    @property
    def cached_len(self) -> int:
        return self._cached_len
    
    def complete_one(self, next_token: torch.Tensor) -> None:
        # complete one step, either prefill or decode
        self.append_host(next_token)
        self._cached_len = self.device_len
        self.device_len += 1
        
    def append_host(self, next_token: torch.Tensor) -> None:
        # append new token to the token buffer on host
        self.host_ids = torch.cat([self.host_ids, next_token.unsqueeze(0)])
        
    def can_decode(self) -> bool:
        return self.remain_len > 0
        
    def __repr__(self) -> str:
        return (
            f"{type(self)}(uid={self.uid}, "
            f"device_len={self.device_len}, max_device_len={self.max_device_len}, "
            f"remain_len={self.remain_len}, extend_len={self.extend_len})"
        )
        
        
@dataclass
class Batch:
    # batch is a group of requests with different lengths
    def __init__(self, *, reqs: List[Req], phase: Literal["prefill", "decode"]):
        self.reqs: List[Req] = reqs
        self.padded_reqs: List[Req] = [] # may contain some dummy reqs for padding
        self.phase: Literal["prefill", "decode"] = phase
        # this fields should be set by attention backend
        self.attn_backend: BaseAttnBackend
        
        # input token ids for prefill
        self.input_ids: torch.Tensor
        # indices of token to prefill 
        self.load_indices: torch.Tensor
        # indices of new token to generate
        self.write_indices: torch.Tensor
        
        
    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        return len(self.padded_reqs)

@dataclass
class BaseAttnMetadata:
    positions: torch.Tensor

class BaseAttnBackend(ABC):
    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor: ...

    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None: ...
    
    @abstractmethod
    def get_attn_metadata(self, batch: Batch) -> BaseAttnMetadata: ...