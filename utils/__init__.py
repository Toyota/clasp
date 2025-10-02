import sys  
import os
# Add parent directory to path if needed
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pandas as pd
import torch  
from torch.utils.data import random_split  
from torch_geometric.loader import DataLoader  
from typing import Tuple, Optional
from dataloaders.common import generate_full_path, seed_worker  

def load_metadata_and_embeddings(load_path: str, cod_basepath: str = "/cif") -> pd.DataFrame:
    """Load crystal metadata and generate full CIF file paths.
    
    Args:
        load_path: Path to the metadata CSV file.
        cod_basepath: Base directory path for CIF files. Defaults to "/cif".
        
    Returns:
        pd.DataFrame: DataFrame containing metadata with added 'cif_path' column.
        
    Raises:
        FileNotFoundError: If the metadata file doesn't exist.
        ValueError: If required columns are missing.
    """  
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Metadata file not found: {load_path}")
        
    metadata_df = pd.read_csv(load_path)  
    
    if 'title' not in metadata_df.columns:
        raise ValueError("Metadata must contain 'title' column")
    if 'file' not in metadata_df.columns:
        raise ValueError("Metadata must contain 'file' column")
    
    metadata_df = metadata_df.dropna(subset=['title'])
    metadata_df['cif_path'] = metadata_df['file'].astype(str).apply(lambda x: generate_full_path(x, base_path=cod_basepath))
    return metadata_df  
  

def prepare_data_loaders(
    batch_size: int, 
    dataset, 
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    num_workers: int = 8,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Prepare data loaders for train, validation and test sets.
    
    Args:
        batch_size: Number of samples per batch.
        dataset: The dataset to split.
        train_ratio: Proportion of data for training. Defaults to 0.8.
        val_ratio: Proportion of data for validation. Defaults to 0.1.
        num_workers: Number of worker processes for data loading. Defaults to 8.
        seed: Random seed for reproducible splits. Defaults to 42.
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test data loaders.
        
    Raises:
        ValueError: If batch_size <= 0 or ratios are invalid.
    """  
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    
    if train_ratio < 0 or train_ratio > 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    
    if val_ratio < 0 or val_ratio > 1:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    
    if train_ratio + val_ratio > 1:
        raise ValueError(f"train_ratio + val_ratio must be <= 1, got {train_ratio + val_ratio}")
    
    # Split dataset into train, validation and test  
    dataset_size = len(dataset)  
    train_size = int(train_ratio * dataset_size)  
    val_size = int(val_ratio * dataset_size)  
    test_size = dataset_size - train_size - val_size  
  
    generator = torch.Generator().manual_seed(seed)  
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)  
  
    # Create data loaders  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True,
                              pin_memory=True, persistent_workers=True,
                              worker_init_fn=seed_worker)  
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False,
                            pin_memory=True,persistent_workers=True,
                            worker_init_fn=seed_worker)  
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False,
                             pin_memory=True,persistent_workers=True,
                             worker_init_fn=seed_worker)  
  
    return train_loader, val_loader, test_loader  