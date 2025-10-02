"""Unit tests for CLaSP dataloaders."""

import unittest
import os
import sys
import tempfile
import pandas as pd
import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.common import generate_full_path, exclude_one_atom_crystal
from torch_geometric.data import Data


class TestDataLoaderUtils(unittest.TestCase):
    """Test dataloader utility functions."""

    def test_generate_full_path(self):
        """Test path generation for CIF files."""
        # Test standard case
        self.assertEqual(
            generate_full_path("1000001", base_path="/cod"),
            "/cod/1/00/00/1000001.cif"
        )
        
        # Test with different base path
        self.assertEqual(
            generate_full_path("2345678", base_path="/data/cif"),
            "/data/cif/2/34/56/2345678.cif"
        )
        
        # Test minimum length case
        self.assertEqual(
            generate_full_path("123456", base_path="/cod"),
            "/cod/1/23/45/123456.cif"
        )
        
        # Test error case for short filename
        with self.assertRaises(ValueError):
            generate_full_path("12345", base_path="/cod")

    def test_exclude_one_atom_crystal(self):
        """Test filtering of single-atom crystals."""
        # Create test data with 1 atom
        data_single = Data(
            x=torch.tensor([[1.0, 2.0]]),
            edge_index=torch.tensor([[], []], dtype=torch.long),
            pos=torch.tensor([[0.0, 0.0, 0.0]])
        )
        
        # Create test data with multiple atoms
        data_multi = Data(
            x=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            pos=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        )
        
        # Test filtering
        self.assertFalse(exclude_one_atom_crystal(data_single))
        self.assertTrue(exclude_one_atom_crystal(data_multi))


class TestDataset(unittest.TestCase):
    """Test dataset creation and loading."""

    def setUp(self):
        """Set up test data."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create a simple test DataFrame
        self.test_df = pd.DataFrame({
            'file': ['1000001', '1000002', '1000003'],
            'title': ['Test crystal 1', 'Test crystal 2', 'Test crystal 3'],
            'cif_path': [
                '/path/to/1000001.cif',
                '/path/to/1000002.cif',
                '/path/to/1000003.cif'
            ]
        })

    def tearDown(self):
        """Clean up test directory."""
        import shutil
        shutil.rmtree(self.test_dir)

    def test_dataframe_processing(self):
        """Test DataFrame processing for dataset creation."""
        # Test dropna functionality
        df_with_na = self.test_df.copy()
        df_with_na.loc[1, 'title'] = None
        
        df_cleaned = df_with_na.dropna(subset=['title'])
        self.assertEqual(len(df_cleaned), 2)
        self.assertNotIn(None, df_cleaned['title'].values)


if __name__ == '__main__':
    unittest.main()