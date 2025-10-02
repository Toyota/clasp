"""Unit tests for CLaSP loss functions."""

import unittest
import os
import sys
import torch
import torch.nn.functional as F

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from losses.clip_loss import clip_loss, cosface_loss


class TestLossFunctions(unittest.TestCase):
    """Test loss function implementations."""

    def setUp(self):
        """Set up test data."""
        self.batch_size = 8
        self.embedding_dim = 128
        
        # Create dummy embeddings
        self.crystal_embeddings = F.normalize(torch.randn(self.batch_size, self.embedding_dim), dim=1)
        self.text_embeddings = F.normalize(torch.randn(self.batch_size, self.embedding_dim), dim=1)

    def test_clip_loss(self):
        """Test CLIP loss computation."""
        # Create params object
        params = type('Params', (), {'loss_scale': 1.0})()
        
        loss = clip_loss(self.text_embeddings, self.crystal_embeddings, params)
        
        # Check loss is scalar
        self.assertEqual(loss.shape, torch.Size([]))
        
        # Check loss is positive
        self.assertGreater(loss.item(), 0)
        
        # Test with identical embeddings (should give lower loss)
        identical_loss = clip_loss(self.crystal_embeddings, self.crystal_embeddings, params)
        self.assertLess(identical_loss.item(), loss.item())

    def test_cosface_loss(self):
        """Test CosFace loss computation."""
        # Parameters for CosFace
        params = type('Params', (), {'loss_scale': 64.0, 'margin': 0.35})()
        
        loss = cosface_loss(self.text_embeddings, self.crystal_embeddings, params)
        
        # Check loss is scalar
        self.assertEqual(loss.shape, torch.Size([]))
        
        # Check loss is positive
        self.assertGreater(loss.item(), 0)
        
        # Test with different margins
        params_small = type('Params', (), {'loss_scale': 64.0, 'margin': 0.1})()
        params_large = type('Params', (), {'loss_scale': 64.0, 'margin': 0.5})()
        
        loss_small_margin = cosface_loss(self.text_embeddings, self.crystal_embeddings, params_small)
        loss_large_margin = cosface_loss(self.text_embeddings, self.crystal_embeddings, params_large)
        
        # Larger margin should generally give larger loss
        self.assertLess(loss_small_margin.item(), loss_large_margin.item())



class TestLossUtils(unittest.TestCase):
    """Test loss utility functions."""

    def test_similarity_computation(self):
        """Test similarity matrix computation."""
        # Create normalized embeddings
        emb1 = F.normalize(torch.randn(4, 32), dim=1)
        emb2 = F.normalize(torch.randn(4, 32), dim=1)
        
        # Compute similarity
        sim = emb1 @ emb2.t()
        
        # Check shape
        self.assertEqual(sim.shape, (4, 4))
        
        # Check values are in [-1, 1] (cosine similarity range)
        self.assertTrue(torch.all(sim >= -1.0))
        self.assertTrue(torch.all(sim <= 1.0))
        
        # Check diagonal for self-similarity
        self_sim = emb1 @ emb1.t()
        diagonal = torch.diag(self_sim)
        torch.testing.assert_close(diagonal, torch.ones(4), rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()