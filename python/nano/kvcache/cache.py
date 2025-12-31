from dataclasses import dataclass
import enum
from abc import ABC, abstractmethod
from typing import Tuple, NamedTuple
import torch

@dataclass
class KVCache:
    
    def __init__(self, 
                 num_pages: int, 
                 num_kv_heads: int, 
                 num_layers: int, 
                 head_dim: int, 
                 dtype: torch.dtype, 
                 device: torch.device,
        ):
        self.num_kv_heads = num_kv_heads
        self._num_layers = num_layers
        self.head_dim = head_dim
        self.num_pages = num_pages
        self._dtype = dtype
        self._device = device


        # TODO: implement head split with tensor parallel
        local_kv_heads = self.num_kv_heads
        self.kv_buffer = torch.empty(
            (2, num_layers, num_pages, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self.k_buffer = self.kv_buffer[0]
        self.v_buffer = self.kv_buffer[1]
        self.storage_shape = (num_pages, local_kv_heads, head_dim)
        
    def k_cache(self, index: int) -> torch.Tensor:
        return self.k_buffer[index]
    
    def v_cache(self, index: int) -> torch.Tensor:
        return self.v_buffer[index]
    
    def store_kv(self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int) -> None:
        """ Store key and value tensors to the cache at the specified layer and output locations.
        """
        k_cache = self.k_buffer[layer_id].view(self.storage_shape)
        v_cache = self.v_buffer[layer_id].view(self.storage_shape)
        k_cache[out_loc, :] = k
        v_cache[out_loc, :] = v
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    @property
    def dtype(self) -> torch.dtype:
        return self._dtype
    
    @property
    def num_layers(self) -> int:
        return self._num_layers

@dataclass(frozen=True)
class CacheHandle:
    cached_len: int
    
    
class SizeInfo(NamedTuple):
    evictable_size: int
    protected_size: int

    @property
    def total_size(self) -> int:
        return self.evictable_size + self.protected_size

        
class KVCacheManager(ABC):
    """
    Cache manager interface for KVCache.
    """
    
    @abstractmethod
    def match_prefix(self, input_ids: torch.Tensor) -> Tuple[CacheHandle, torch.Tensor]:
        """
        Match prefix and return the indices of the matched prefix in the cache.
        This operation will not modify the cache.
        The returned indices is only safe to use when the handle is locked.

        Args:
            input_ids (torch.Tensor): The input ids to match. Shape: (seq_len,)
        Returns:
            handle (BaseCacheHandle): The handle to the matched prefix.
            indices (torch.Tensor): The indices of the longest-matched prefix in the cache.
        """

    @abstractmethod
    def lock_handle(self, handle: CacheHandle, unlock: bool = False) -> None:
        """
        Lock or unlock a cache handle.
        This operation will not modify the cache, but change the size info only.
        When a handle is locked, it cannot be evicted.
        Handles must be locked before the previously-returned tensor of `match_prefix` is used.
        Otherwise it may be evicted by calling evict.

        Args:
            handle (BaseCacheHandle): The cache handle to lock or unlock.
            unlock (bool): Whether to unlock the handle. Defaults to False.
        """

    @abstractmethod
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> int:
        """
        Insert a new prefix into the cache.
        This operation will modify the cache.
        Args:
            input_ids (torch.Tensor): The input ids to insert. Shape: (seq_len,)
            indices (torch.Tensor): The indices to store the new prefix. Shape: (seq_len,)

        Returns:
            int: The length of prefix that is already in the cache. This part is not
                 inserted, so the caller should free these indices.
        """

    @abstractmethod
    def evict(self, size: int) -> torch.Tensor:
        """
        Evict some prefixes from the cache to free up space.
        This operation will modify the cache.
        Note that evict 0 is always safe and does nothing.
        Note that the actual evict size may be larger than the requested size.
        Args:
            size (int): The size to evict.

        Returns:
            torch.Tensor: The indices evicted. Shape: (evict_size,)
        Raises:
            RuntimeError: If the requested size is larger than the evictable size.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the cache manager and the underlying cache."""

    @property
    @abstractmethod
    def size_info(self) -> SizeInfo:
        """Get the size information of the cache."""

    @abstractmethod
    def check_integrity(self) -> None:
        """Check the integrity of the cache. Raise an error if the cache is corrupted."""
        
    