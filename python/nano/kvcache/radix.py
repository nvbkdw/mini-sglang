from typing import Optional, Tuple, NamedTuple
from dataclasses import dataclass
from typing import Dict, List
import time
import heapq
import torch
from nano.kvcache.cache import KVCacheManager, CacheHandle

class RadixTreeNode:
    counter: int = 0
    
    def __init__(self, tic: int | None = None) -> None:
        self.children: Dict[int, RadixTreeNode] = {}
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1
        self.timestamp = tic or time.monotonic_ns()
        
        self._key: torch.Tensor # token IDs for this node
        self._value: torch.Tensor # page indices in KV cache
        self._length: int # length of tokens
        
    def set_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
        assert len(key) == len(value)
        self._key = key
        self._value = value
        self._length = len(key)
        
    def set_parent(self, parent: 'RadixTreeNode') -> None:
        self._parent = parent
        parent.children[int(self._key[0].item())] = self
        
    @property
    def length(self) -> int:
        return self._length
    
    @property
    def parent(self) -> 'RadixTreeNode':
        assert self._parent is not None
        return self._parent
    
    @property
    def value(self) -> torch.Tensor:
        return self._value
    
    def is_root(self) -> bool:
        return self._parent is None
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def get_match_len(self, input_ids: torch.Tensor) -> int:
        """ get the length of the longest prefix match between the key and the input_ids
        """
        max_len = min(len(self._key), len(input_ids))
        for i in range(max_len):
            if self._key[i] != input_ids[i]:
                return i
        return max_len
    
    def __lt__(self, other: 'RadixTreeNode') -> bool:
        """Compare nodes by timestamp for heap operations (LRU eviction)."""
        return self.timestamp < other.timestamp
    
    def _split_at(self, pos: int) -> 'RadixTreeNode':
        """ split the node at the given position
        """
        assert 0 < pos < self.length
        parent = self.parent
       
       # create a new node for [pos:]
        new_node = RadixTreeNode(tic=self.timestamp)
        new_node.set_key_value(self._key[pos:], self._value[pos:])
        new_node.set_parent(self)
        new_node.ref_count = self.ref_count
        new_node.children = self.children.copy()
        
        # update current node to [pos:]
        self.set_key_value(self._key[:pos], self._value[:pos])
        self.children = {}
        self.children[int(new_node._key[0].item())] = new_node
        return self

@dataclass(frozen=True)
class RadixCacheHandle(CacheHandle):
    node: RadixTreeNode
    
class SizeInfo(NamedTuple):
    evictable_size: int
    protected_size: int

    @property
    def total_size(self) -> int:
        return self.evictable_size + self.protected_size

class RadixKVCacheManager(KVCacheManager):
    
    def __init__(self):
        self._empty_tensor: Optional[torch.Tensor] = torch.empty(0, dtype=torch.int32)
        super().__init__()
        
        self.root_node = RadixTreeNode()
        self.root_node.ref_count = 1  # root is always protected
        self.evictable_size = 0
        self.protected_size = 0
        
    def lock_handle(self, handle: CacheHandle, unlock: bool = False) -> None:
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        if unlock:
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
                node = node.parent
        
    def match_prefix(self, input_ids: torch.Tensor) -> Tuple[RadixCacheHandle, torch.Tensor]:
        node, prefix_len = self._walk(input_ids)
        if prefix_len == 0:
            assert node.is_root() and node is self.root_node and prefix_len == 0
            return RadixCacheHandle(cached_len=0, node=node), self._empty_tensor
        matched_node = node  # Save the matched node before walking to root
        value_list: List[torch.Tensor] = []
        while not node.is_root():
            value_list.append(node.value)
            node = node.parent
        value_list.reverse()
        return RadixCacheHandle(prefix_len, matched_node), torch.cat(value_list)
        
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> int:
        node, prefix_len = self._walk(input_ids)
        assert prefix_len <= len(input_ids)
        # allocate new node to store the remaining part of the input_ids
        if prefix_len < len(input_ids):
            new_node = RadixTreeNode()
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:])
            new_node.set_parent(node)
            # leaf node is evictable
            self.evictable_size += new_node.length
        return prefix_len
            
    def evict(self, size: int) -> torch.Tensor:
        
        if size == 0:
            return self._empty_tensor
        
        leave_nodes = self._collect_leave_nodes_for_evict()
        heapq.heapify(leave_nodes)  # min-heap by timestamp (oldest first)
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0
        
        while evicted_size < size:
            node = heapq.heappop(leave_nodes)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.length
            evicted_indices.append(node.value)
            self.evictable_size -= node.length
            parent = node.parent
            del parent.children[int(node._key[0].item())]
            # NOTE: root is always protected, so won't be evicted
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)
        
        return torch.cat(evicted_indices)
        
    def reset(self) -> None:
        raise NotImplementedError("RadixKVCacheManager.reset is not implemented")
        
    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(
            evictable_size=self.evictable_size, 
            protected_size=self.protected_size
        )
        
    def check_integrity(self) -> None:
        raise NotImplementedError("RadixKVCacheManager.check_integrity is not implemented")
        
    def _walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        """ Walk the radix tree to find the longest prefix match.
        """
        
        # start from root
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()
        
        # traverse the tree by token ID
        while prefix_len < indice_len:
            this_id = int(input_ids[prefix_len].item())
            if this_id not in node.children:
                return node, prefix_len
            
            node = node.children[this_id]
            
            # NOTE: at least 1 char is matched, so match_len >= 1
            match_len = node.get_match_len(input_ids[prefix_len:])
            prefix_len += match_len
        
            # partial match, split the node at the mismatch position
            if match_len != node.length:
                node = node._split_at(match_len)
                return node, prefix_len
            
            node.timestamp = tic
            
        return node, prefix_len
    
    def _collect_leave_nodes_for_evict(self) -> List[RadixTreeNode]:
        """ collect all leave nodes that are evictable
        """
        nodes: List[RadixTreeNode] = [self.root_node]
        leave_nodes: List[RadixTreeNode] = []
        
        while len(nodes) > 0:
            node = nodes.pop()
            if node.is_leaf():
                if node.ref_count == 0:
                    leave_nodes.append(node)
            else:
                for child in node.children.values():
                    nodes.append(child)
                    
        return leave_nodes
    
    