#!/usr/bin/env python
"""
Extract CLaSP embeddings from CIF files.

Usage:
    python extract_embeddings.py \
        --checkpoint_path /path/to/checkpoint.ckpt \
        --cif_list /path/to/cif_list.txt \
        --output_path embeddings.npz \
        --batch_size 32
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch_geometric.loader import DataLoader

# Add CLaSP to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from models.contrastive import ClaspModel
from dataloaders.dataset import ClaspDataset
from models.utils import normalize_embedding


def parse_args():
    parser = argparse.ArgumentParser(description='Extract CLaSP embeddings from CIF files.')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to CLaSP checkpoint file')
    parser.add_argument('--cif_list', type=str, required=True,
                        help='Path to text file containing list of CIF file paths')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save output embeddings (npz file)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing (default: 32)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to config directory (default: auto-detect)')
    parser.add_argument('--config_name', type=str, default='soroban_dev',
                        help='Config name to use (default: soroban_dev)')
    return parser.parse_args()


def load_cif_paths(cif_list_path):
    """Load CIF file paths from text file."""
    with open(cif_list_path, 'r') as f:
        cif_paths = [line.strip() for line in f if line.strip()]
    return cif_paths


def create_dataframe(cif_paths):
    """Create DataFrame from CIF paths."""
    data = []
    for cif_path in cif_paths:
        cif_filename = Path(cif_path).stem
        data.append({
            'file': cif_filename,
            'title': f'Crystal structure {cif_filename}',  # Dummy title
            'cif_path': cif_path
        })
    return pd.DataFrame(data)


def extract_embeddings(model, dataloader, device, normalize_mode=None):
    """Extract embeddings from dataloader."""
    embeddings = []
    file_ids = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            batch = batch.to(device)
            
            # Get crystal embedding
            crystal_emb = model.model(batch)
            
            # Normalize if configured
            if normalize_mode:
                crystal_emb = normalize_embedding(crystal_emb, normalize_mode)
            
            embeddings.append(crystal_emb.cpu().numpy())
            
            # Store file IDs
            for i in range(len(batch)):
                file_ids.append(batch.material_id[i].item())
    
    embeddings = np.vstack(embeddings)
    return embeddings, file_ids


def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load CIF paths
    print("Loading CIF paths...")
    cif_paths = load_cif_paths(args.cif_list)
    print(f"Found {len(cif_paths)} CIF files")
    
    # Create DataFrame
    df = create_dataframe(cif_paths)
    
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    # Load configuration
    print("Loading configuration...")
    # Change to directory containing the script to make config path relative
    original_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    config_path = args.config_path or '../configs'
    with initialize(config_path=config_path, version_base='1.1'):
        cfg = compose(config_name=args.config_name)
    
    # Override some settings for inference
    cfg.freeze_text_encoders = True
    
    print(f"Configuration loaded:")
    print(f"  - Embedding dim: {cfg.embedding_dim}")
    print(f"  - Normalize mode: {cfg.embedding_normalize}")
    
    # Load model from checkpoint
    print(f"Loading model from {args.checkpoint_path}...")
    model = ClaspModel.load_from_checkpoint(
        args.checkpoint_path,
        cfg=cfg,
        train_loader=None,
        val_loader=None,
        map_location=device,
        strict=False
    )
    model.eval()
    model.to(device)
    print("Model loaded successfully")
    
    # Create dataset
    print("Creating dataset...")
    # Use dummy tokenizer since we're only extracting crystal embeddings
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_textencoder_model_id)
    
    dataset = ClaspDataset(
        input_dataframe=df,
        tokenizer=tokenizer,
        max_token_length=cfg.max_token_length,
        root='temp_cache_embeddings'
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0  # Avoid multiprocessing issues
    )
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings, file_ids = extract_embeddings(
        model, 
        dataloader, 
        device, 
        cfg.embedding_normalize
    )
    
    # Create mapping from file_ids to original filenames
    filenames = [df.iloc[i]['file'] for i in file_ids]
    
    # Change back to original directory before saving
    os.chdir(original_dir)
    
    # Save embeddings
    print(f"Saving embeddings to {args.output_path}...")
    np.savez(
        args.output_path,
        embeddings=embeddings,
        filenames=filenames,
        cif_paths=[df.iloc[i]['cif_path'] for i in file_ids]
    )
    
    print(f"Successfully saved embeddings for {len(embeddings)} structures")
    print(f"Embedding shape: {embeddings.shape}")


if __name__ == '__main__':
    main()