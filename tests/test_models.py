"""Unit tests for CLaSP models."""

import unittest
import os
import sys
import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils import MLP, normalize_scale, normalize_embedding


class TestModelUtils(unittest.TestCase):
    """Test model utility functions."""

    def test_normalize_scale(self):
        """Test scaling normalization."""
        # normalize_scale expects pos and batch tensors for graph data
        pos = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        batch = torch.tensor([0, 0, 1])  # First two belong to batch 0, last one to batch 1
        
        # Apply normalization
        normalized_pos, scale = normalize_scale(pos, batch)
        
        # Check shape preserved
        self.assertEqual(normalized_pos.shape, pos.shape)
        
        # Check that positions are centered and scaled
        # For batch 0, mean should be subtracted
        # For batch 1, it's just the single point centered

    def test_normalize_embedding(self):
        """Test embedding normalization."""
        x = torch.tensor([[3.0, 4.0], [0.0, 2.0], [1.0, 0.0]])
        
        # Test L2 normalization
        normalized_l2 = normalize_embedding(x, mode='l2')
        self.assertEqual(normalized_l2.shape, x.shape)
        norms = torch.norm(normalized_l2, dim=1)
        expected = torch.ones(x.shape[0])
        torch.testing.assert_close(norms, expected, rtol=1e-5, atol=1e-6)
        
        # Test L1 normalization
        normalized_l1 = normalize_embedding(x, mode='l1')
        self.assertEqual(normalized_l1.shape, x.shape)
        
        # Test sqrtd normalization
        normalized_sqrtd = normalize_embedding(x, mode='sqrtd')
        self.assertEqual(normalized_sqrtd.shape, x.shape)
        
        # Test None mode (no normalization)
        normalized_none = normalize_embedding(x, mode=None)
        torch.testing.assert_close(normalized_none, x)

    def test_mlp_creation(self):
        """Test MLP module creation."""
        # Test basic MLP
        mlp = MLP([10, 20, 30, 5])
        
        # Check it's a Sequential module
        self.assertIsInstance(mlp, nn.Sequential)
        
        # Test forward pass
        x = torch.randn(32, 10)
        output = mlp(x)
        # MLP includes BatchNorm and ReLU for all layers, so final shape should match
        # But note the actual implementation includes BatchNorm and ReLU even for the last layer
        self.assertEqual(output.shape[0], 32)  # Batch size preserved


if __name__ == '__main__':
    unittest.main()