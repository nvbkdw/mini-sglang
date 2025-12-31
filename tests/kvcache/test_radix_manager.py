"""
Unit tests for RadixCacheManager.

This module tests the radix tree-based cache manager which implements
prefix caching with LRU eviction.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from minisgl.kvcache.radix_manager import (
    RadixCacheHandle,
    RadixCacheManager,
    RadixTreeNode,
)
from minisgl.utils import call_if_main


def mock_fast_compare_key(key: torch.Tensor, input_ids: torch.Tensor) -> int:
    """Pure Python implementation of fast_compare_key for testing."""
    min_len = min(len(key), len(input_ids))
    for i in range(min_len):
        if key[i].item() != input_ids[i].item():
            return i
    return min_len


class TestRadixTreeNode(unittest.TestCase):
    """Test cases for RadixTreeNode class."""

    def test_node_creation(self):
        """Test basic node creation."""
        node = RadixTreeNode()
        self.assertEqual(node.ref_count, 0)
        self.assertEqual(len(node.children), 0)
        self.assertTrue(node.is_root())  # No parent set
        self.assertTrue(node.is_leaf())  # No children

    def test_set_key_value(self):
        """Test setting key and value on a node."""
        node = RadixTreeNode()
        key = torch.tensor([1, 2, 3], dtype=torch.int32)
        value = torch.tensor([10, 20, 30], dtype=torch.int32)
        node.set_key_value(key, value)

        self.assertEqual(node.length, 3)
        self.assertTrue(torch.equal(node.value, value))

    def test_set_parent(self):
        """Test parent-child relationship."""
        parent = RadixTreeNode()
        parent.set_key_value(torch.tensor([0], dtype=torch.int32), torch.tensor([0], dtype=torch.int32))
        
        child = RadixTreeNode()
        child.set_key_value(torch.tensor([5, 6], dtype=torch.int32), torch.tensor([50, 60], dtype=torch.int32))
        child.set_parent(parent)

        self.assertFalse(child.is_root())
        self.assertEqual(child.parent, parent)
        self.assertIn(5, parent.children)
        self.assertEqual(parent.children[5], child)
        self.assertFalse(parent.is_leaf())

    def test_node_comparison(self):
        """Test node comparison for heap operations (LRU)."""
        node1 = RadixTreeNode(tic=100)
        node2 = RadixTreeNode(tic=200)
        
        self.assertTrue(node1 < node2)
        self.assertFalse(node2 < node1)

    def test_split_at(self):
        """Test splitting a node at a position."""
        parent = RadixTreeNode()
        parent.set_key_value(torch.tensor([0], dtype=torch.int32), torch.tensor([0], dtype=torch.int32))
        
        node = RadixTreeNode()
        key = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
        value = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32)
        node.set_key_value(key, value)
        node.set_parent(parent)
        node.ref_count = 2

        # Split at position 2
        new_node = node._split_at(2)

        # Check new node (prefix)
        self.assertEqual(new_node.length, 2)
        self.assertTrue(torch.equal(new_node.value, torch.tensor([10, 20], dtype=torch.int32)))
        self.assertEqual(new_node.parent, parent)
        self.assertEqual(new_node.ref_count, 2)

        # Check original node (suffix)
        self.assertEqual(node.length, 3)
        self.assertTrue(torch.equal(node.value, torch.tensor([30, 40, 50], dtype=torch.int32)))
        self.assertEqual(node.parent, new_node)

        # Check parent references
        self.assertEqual(parent.children[1], new_node)
        self.assertEqual(new_node.children[3], node)


class TestRadixCacheManager(unittest.TestCase):
    """Test cases for RadixCacheManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.patcher = patch(
            "minisgl.kvcache.radix_manager.RadixTreeNode.get_match_len",
            side_effect=lambda self, input_ids: mock_fast_compare_key(self._key, input_ids),
        )
        self.patcher.start()

    def tearDown(self):
        """Clean up after tests."""
        self.patcher.stop()

    def test_initialization(self):
        """Test manager initialization."""
        manager = RadixCacheManager(self.device)
        
        self.assertEqual(manager.evictable_size, 0)
        self.assertEqual(manager.protected_size, 0)
        self.assertIsNotNone(manager.root_node)
        self.assertEqual(manager.root_node.ref_count, 1)  # Root is always protected

    def test_size_info(self):
        """Test size_info property."""
        manager = RadixCacheManager(self.device)
        
        size_info = manager.size_info
        self.assertEqual(size_info.evictable_size, 0)
        self.assertEqual(size_info.protected_size, 0)
        self.assertEqual(size_info.total_size, 0)

    def test_match_prefix_empty_tree(self):
        """Test prefix matching on empty tree."""
        manager = RadixCacheManager(self.device)
        input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
        
        handle, indices = manager.match_prefix(input_ids)
        
        self.assertEqual(handle.cached_len, 0)
        self.assertEqual(len(indices), 0)
        self.assertTrue(handle.node.is_root())

    def test_insert_prefix_single(self):
        """Test inserting a single prefix."""
        manager = RadixCacheManager(self.device)
        input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
        indices = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32)
        
        prefix_len = manager.insert_prefix(input_ids, indices)
        
        self.assertEqual(prefix_len, 0)  # Nothing was cached before
        self.assertEqual(manager.evictable_size, 5)

    def test_match_prefix_after_insert(self):
        """Test prefix matching after inserting."""
        manager = RadixCacheManager(self.device)
        input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
        indices = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32)
        
        manager.insert_prefix(input_ids, indices)
        
        # Match exact prefix
        handle, matched_indices = manager.match_prefix(input_ids)
        self.assertEqual(handle.cached_len, 5)
        self.assertTrue(torch.equal(matched_indices, indices))

    def test_match_prefix_partial(self):
        """Test partial prefix matching."""
        manager = RadixCacheManager(self.device)
        input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
        indices = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32)
        
        manager.insert_prefix(input_ids, indices)
        
        # Match shorter prefix
        query = torch.tensor([1, 2, 3], dtype=torch.int32)
        handle, matched_indices = manager.match_prefix(query)
        self.assertEqual(handle.cached_len, 3)
        self.assertTrue(torch.equal(matched_indices, torch.tensor([10, 20, 30], dtype=torch.int32)))

    def test_match_prefix_longer_query(self):
        """Test prefix matching with longer query."""
        manager = RadixCacheManager(self.device)
        input_ids = torch.tensor([1, 2, 3], dtype=torch.int32)
        indices = torch.tensor([10, 20, 30], dtype=torch.int32)
        
        manager.insert_prefix(input_ids, indices)
        
        # Query longer than cached
        query = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
        handle, matched_indices = manager.match_prefix(query)
        self.assertEqual(handle.cached_len, 3)
        self.assertTrue(torch.equal(matched_indices, indices))

    def test_match_prefix_no_match(self):
        """Test prefix matching with no match."""
        manager = RadixCacheManager(self.device)
        input_ids = torch.tensor([1, 2, 3], dtype=torch.int32)
        indices = torch.tensor([10, 20, 30], dtype=torch.int32)
        
        manager.insert_prefix(input_ids, indices)
        
        # Query with different start
        query = torch.tensor([5, 6, 7], dtype=torch.int32)
        handle, matched_indices = manager.match_prefix(query)
        self.assertEqual(handle.cached_len, 0)
        self.assertEqual(len(matched_indices), 0)

    def test_insert_prefix_with_overlap(self):
        """Test inserting prefixes with overlap."""
        manager = RadixCacheManager(self.device)
        
        # Insert first prefix
        input_ids1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
        indices1 = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32)
        prefix_len1 = manager.insert_prefix(input_ids1, indices1)
        self.assertEqual(prefix_len1, 0)
        self.assertEqual(manager.evictable_size, 5)
        
        # Insert overlapping prefix
        input_ids2 = torch.tensor([1, 2, 3, 6, 7], dtype=torch.int32)
        indices2 = torch.tensor([10, 20, 30, 60, 70], dtype=torch.int32)
        prefix_len2 = manager.insert_prefix(input_ids2, indices2)
        self.assertEqual(prefix_len2, 3)  # 3 tokens already cached
        self.assertEqual(manager.evictable_size, 7)  # 5 + 2 new tokens

    def test_insert_duplicate_prefix(self):
        """Test inserting duplicate prefix."""
        manager = RadixCacheManager(self.device)
        input_ids = torch.tensor([1, 2, 3], dtype=torch.int32)
        indices = torch.tensor([10, 20, 30], dtype=torch.int32)
        
        manager.insert_prefix(input_ids, indices)
        prefix_len = manager.insert_prefix(input_ids, indices)
        
        self.assertEqual(prefix_len, 3)  # All already cached
        self.assertEqual(manager.evictable_size, 3)  # No new tokens

    def test_lock_handle(self):
        """Test locking a handle."""
        manager = RadixCacheManager(self.device)
        input_ids = torch.tensor([1, 2, 3], dtype=torch.int32)
        indices = torch.tensor([10, 20, 30], dtype=torch.int32)
        
        manager.insert_prefix(input_ids, indices)
        self.assertEqual(manager.evictable_size, 3)
        self.assertEqual(manager.protected_size, 0)
        
        # Lock the handle
        handle, _ = manager.match_prefix(input_ids)
        manager.lock_handle(handle)
        
        self.assertEqual(manager.evictable_size, 0)
        self.assertEqual(manager.protected_size, 3)

    def test_unlock_handle(self):
        """Test unlocking a handle."""
        manager = RadixCacheManager(self.device)
        input_ids = torch.tensor([1, 2, 3], dtype=torch.int32)
        indices = torch.tensor([10, 20, 30], dtype=torch.int32)
        
        manager.insert_prefix(input_ids, indices)
        
        handle, _ = manager.match_prefix(input_ids)
        manager.lock_handle(handle)
        manager.lock_handle(handle, unlock=True)
        
        self.assertEqual(manager.evictable_size, 3)
        self.assertEqual(manager.protected_size, 0)

    def test_evict_zero(self):
        """Test evicting zero size."""
        manager = RadixCacheManager(self.device)
        
        evicted = manager.evict(0)
        self.assertEqual(len(evicted), 0)

    def test_evict_simple(self):
        """Test simple eviction."""
        manager = RadixCacheManager(self.device)
        input_ids = torch.tensor([1, 2, 3], dtype=torch.int32)
        indices = torch.tensor([10, 20, 30], dtype=torch.int32)
        
        manager.insert_prefix(input_ids, indices)
        self.assertEqual(manager.evictable_size, 3)
        
        evicted = manager.evict(3)
        
        self.assertEqual(len(evicted), 3)
        self.assertTrue(torch.equal(evicted, indices))
        self.assertEqual(manager.evictable_size, 0)

    def test_evict_partial(self):
        """Test partial eviction."""
        manager = RadixCacheManager(self.device)
        
        # Insert two separate prefixes
        input_ids1 = torch.tensor([1, 2, 3], dtype=torch.int32)
        indices1 = torch.tensor([10, 20, 30], dtype=torch.int32)
        manager.insert_prefix(input_ids1, indices1)
        
        input_ids2 = torch.tensor([5, 6, 7], dtype=torch.int32)
        indices2 = torch.tensor([50, 60, 70], dtype=torch.int32)
        manager.insert_prefix(input_ids2, indices2)
        
        self.assertEqual(manager.evictable_size, 6)
        
        # Evict only 3
        evicted = manager.evict(3)
        
        self.assertEqual(len(evicted), 3)
        self.assertEqual(manager.evictable_size, 3)

    def test_evict_lru_order(self):
        """Test that eviction follows LRU order."""
        manager = RadixCacheManager(self.device)
        
        # Insert first prefix (older)
        input_ids1 = torch.tensor([1, 2, 3], dtype=torch.int32)
        indices1 = torch.tensor([10, 20, 30], dtype=torch.int32)
        manager.insert_prefix(input_ids1, indices1)
        
        # Insert second prefix (newer)
        input_ids2 = torch.tensor([5, 6, 7], dtype=torch.int32)
        indices2 = torch.tensor([50, 60, 70], dtype=torch.int32)
        manager.insert_prefix(input_ids2, indices2)
        
        # Access first prefix to make it newer
        manager.match_prefix(input_ids1)
        
        # Evict - should evict second prefix first (it's now older)
        evicted = manager.evict(3)
        
        self.assertTrue(torch.equal(evicted, indices2))

    def test_evict_protected_fails(self):
        """Test that evicting protected nodes fails."""
        manager = RadixCacheManager(self.device)
        input_ids = torch.tensor([1, 2, 3], dtype=torch.int32)
        indices = torch.tensor([10, 20, 30], dtype=torch.int32)
        
        manager.insert_prefix(input_ids, indices)
        
        # Lock the handle
        handle, _ = manager.match_prefix(input_ids)
        manager.lock_handle(handle)
        
        self.assertEqual(manager.evictable_size, 0)
        
        with self.assertRaises(AssertionError):
            manager.evict(1)

    def test_evict_too_much_fails(self):
        """Test that evicting more than available fails."""
        manager = RadixCacheManager(self.device)
        input_ids = torch.tensor([1, 2, 3], dtype=torch.int32)
        indices = torch.tensor([10, 20, 30], dtype=torch.int32)
        
        manager.insert_prefix(input_ids, indices)
        
        with self.assertRaises(AssertionError):
            manager.evict(10)  # Only 3 available

    def test_multiple_branches(self):
        """Test tree with multiple branches."""
        manager = RadixCacheManager(self.device)
        
        # Insert multiple prefixes sharing a common prefix
        base = torch.tensor([1, 2, 3], dtype=torch.int32)
        base_indices = torch.tensor([10, 20, 30], dtype=torch.int32)
        
        branch1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
        branch1_indices = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32)
        
        branch2 = torch.tensor([1, 2, 3, 6, 7], dtype=torch.int32)
        branch2_indices = torch.tensor([10, 20, 30, 60, 70], dtype=torch.int32)
        
        manager.insert_prefix(branch1, branch1_indices)
        manager.insert_prefix(branch2, branch2_indices)
        
        # Should have total 7 tokens (3 shared + 2 + 2)
        self.assertEqual(manager.evictable_size, 7)
        
        # Match common prefix
        handle, matched = manager.match_prefix(base)
        self.assertEqual(handle.cached_len, 3)
        
        # Match branch1
        handle, matched = manager.match_prefix(branch1)
        self.assertEqual(handle.cached_len, 5)
        
        # Match branch2
        handle, matched = manager.match_prefix(branch2)
        self.assertEqual(handle.cached_len, 5)

    def test_node_splitting(self):
        """Test that nodes are split correctly when inserting partial matches."""
        manager = RadixCacheManager(self.device)
        
        # Insert long prefix
        long_prefix = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int32)
        long_indices = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80], dtype=torch.int32)
        manager.insert_prefix(long_prefix, long_indices)
        
        # Insert prefix that diverges in the middle
        diverging = torch.tensor([1, 2, 3, 4, 9, 10], dtype=torch.int32)
        diverging_indices = torch.tensor([10, 20, 30, 40, 90, 100], dtype=torch.int32)
        prefix_len = manager.insert_prefix(diverging, diverging_indices)
        
        self.assertEqual(prefix_len, 4)  # First 4 tokens matched
        self.assertEqual(manager.evictable_size, 10)  # 8 + 2 new tokens
        
        # Verify both prefixes are accessible
        handle1, matched1 = manager.match_prefix(long_prefix)
        self.assertEqual(handle1.cached_len, 8)
        
        handle2, matched2 = manager.match_prefix(diverging)
        self.assertEqual(handle2.cached_len, 6)

    def test_lock_deep_node(self):
        """Test locking a deeply nested node."""
        manager = RadixCacheManager(self.device)
        
        # Create deep tree structure
        prefix1 = torch.tensor([1, 2, 3], dtype=torch.int32)
        indices1 = torch.tensor([10, 20, 30], dtype=torch.int32)
        manager.insert_prefix(prefix1, indices1)
        
        prefix2 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
        indices2 = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32)
        manager.insert_prefix(prefix2, indices2)
        
        prefix3 = torch.tensor([1, 2, 3, 6, 7], dtype=torch.int32)
        indices3 = torch.tensor([10, 20, 30, 60, 70], dtype=torch.int32)
        manager.insert_prefix(prefix3, indices3)
        
        self.assertEqual(manager.evictable_size, 7)
        
        # Lock a deep handle - should protect entire path
        handle, _ = manager.match_prefix(prefix2)
        manager.lock_handle(handle)
        
        # Only branches not in the locked path should be evictable
        # prefix3 leaf [6,7] is evictable
        self.assertEqual(manager.evictable_size, 2)
        self.assertEqual(manager.protected_size, 5)

    def test_reset_not_implemented(self):
        """Test that reset raises NotImplementedError."""
        manager = RadixCacheManager(self.device)
        
        with self.assertRaises(NotImplementedError):
            manager.reset()

    def test_check_integrity(self):
        """Test check_integrity method (currently no-op)."""
        manager = RadixCacheManager(self.device)
        # Should not raise
        manager.check_integrity()

    def test_radix_cache_handle(self):
        """Test RadixCacheHandle dataclass."""
        node = RadixTreeNode()
        handle = RadixCacheHandle(cached_len=5, node=node)
        
        self.assertEqual(handle.cached_len, 5)
        self.assertEqual(handle.node, node)

    def test_empty_input(self):
        """Test behavior with empty input."""
        manager = RadixCacheManager(self.device)
        input_ids = torch.tensor([], dtype=torch.int32)
        
        handle, indices = manager.match_prefix(input_ids)
        self.assertEqual(handle.cached_len, 0)
        self.assertEqual(len(indices), 0)

    def test_single_token(self):
        """Test with single token sequences."""
        manager = RadixCacheManager(self.device)
        
        input_ids = torch.tensor([42], dtype=torch.int32)
        indices = torch.tensor([100], dtype=torch.int32)
        
        manager.insert_prefix(input_ids, indices)
        
        handle, matched = manager.match_prefix(input_ids)
        self.assertEqual(handle.cached_len, 1)
        self.assertTrue(torch.equal(matched, indices))

    def test_evict_cascades_to_parent(self):
        """Test that eviction properly cascades to make parent a leaf."""
        manager = RadixCacheManager(self.device)
        
        # Insert prefix and extension
        base = torch.tensor([1, 2, 3], dtype=torch.int32)
        base_indices = torch.tensor([10, 20, 30], dtype=torch.int32)
        manager.insert_prefix(base, base_indices)
        
        ext = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
        ext_indices = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32)
        manager.insert_prefix(ext, ext_indices)
        
        self.assertEqual(manager.evictable_size, 5)
        
        # Evict extension first (it's a leaf)
        evicted = manager.evict(2)
        self.assertEqual(len(evicted), 2)
        self.assertEqual(manager.evictable_size, 3)
        
        # Now base becomes a leaf and can be evicted
        evicted = manager.evict(3)
        self.assertEqual(len(evicted), 3)
        self.assertEqual(manager.evictable_size, 0)


@call_if_main(__name__)
def run_tests():
    """Run all unit tests."""
    unittest.main(module="tests.kvcache.test_radix_manager", exit=False, verbosity=2)


if __name__ == "__main__":
    run_tests()

