#!/usr/bin/env python  
# coding: utf-8  
  
import sys  
import os
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd  
import torch  
from torch.utils.data import random_split  
from torch_geometric.loader import DataLoader  
from pytorch_lightning import Trainer, loggers  
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
from omegaconf import DictConfig, OmegaConf  
from transformers import AutoTokenizer
from dataloaders.dataset import ClaspDataset
from dataloaders.common import generate_full_path, seed_worker  
from models.contrastive import ClaspModel  
  
  
def load_metadata_and_embeddings(load_path, cod_basepath="/cif"):  
    """Load the embedding metadata."""  
    metadata_df = pd.read_csv(load_path)  
    metadata_df = metadata_df.dropna(subset=['title'])
    metadata_df['cif_path'] = metadata_df['file'].astype(str).apply(lambda x: generate_full_path(x, base_path=cod_basepath))
    return metadata_df  
  
  
def prepare_data_loaders(cfg, dataset):  
    """Prepare data loaders for train, validation and test."""  
    # Split dataset into train, validation and test  
    dataset_size = len(dataset)  
    train_size = int(0.8 * dataset_size)  
    val_size = int(0.1 * dataset_size)  
    test_size = dataset_size - train_size - val_size  
  
    seed = 42  
    generator = torch.Generator().manual_seed(seed)  
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)  
  
    # Create data loaders  
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=8, drop_last=True,
                              pin_memory=True,persistent_workers=True,
                              worker_init_fn=seed_worker)  
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=8, drop_last=False,
                            pin_memory=True,persistent_workers=True,
                            worker_init_fn=seed_worker)  
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=8, drop_last=False,
                             pin_memory=True,persistent_workers=True,
                             worker_init_fn=seed_worker)  
  
    return train_loader, val_loader, test_loader  
  
  
def setup_trainer(cfg):  
    """Setup the PyTorch Lightning trainer."""  
    print(os.getcwd())
    logger = loggers.TensorBoardLogger(os.getcwd(), name=None)  
    logger.log_hyperparams(cfg)
  
    checkpoint_callback = ModelCheckpoint(save_top_k=cfg.model_checkpoint_save_top_k,  
                                          monitor='val/top01', mode='max', save_last=True,  
                                          dirpath=logger.log_dir+'/model_checkpoint', every_n_epochs=5)  
  
    trainer = Trainer(logger=logger, accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=cfg.num_gpus,  
                      max_epochs=cfg.n_epochs,  
                      default_root_dir=cfg.save_path,  
                      callbacks=[checkpoint_callback],
                      precision="bf16-mixed",
                      log_every_n_steps=cfg.log_every_n_steps,  
                      num_nodes=cfg.num_nodes,  
                      limit_train_batches=cfg.train_percent_check,  
                      limit_val_batches=cfg.val_percent_check,  
                      fast_dev_run=False,  
                      deterministic=False,
                      check_val_every_n_epoch=cfg.check_val_every_n_epoch)  
    return trainer  
  

@hydra.main(version_base=None, config_path="./configs", config_name="training")
def main(cfg : DictConfig):  
  
    # Load embedding metadata  
    metadata_and_embeddings = load_metadata_and_embeddings(load_path=cfg.input_pickle_path,
                                                           cod_basepath=cfg.cod_basepath)
  
    # Display the configuration  
    print(OmegaConf.to_yaml(cfg))  
  
    # Create the dataset  
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_textencoder_model_id)
    dataset = ClaspDataset(input_dataframe=metadata_and_embeddings, 
                            tokenizer=tokenizer, 
                            max_token_length=cfg.max_token_length,
                            root=cfg.dataset_cache_dir)  
    
    del(metadata_and_embeddings)
    del(tokenizer)
  
    # Prepare data loaders for train, validation and test  
    train_loader, val_loader, test_loader = prepare_data_loaders(cfg, dataset)  
  
    # Setup the PyTorch Lightning trainer  
    trainer = setup_trainer(cfg)  
  
    # Initialize the model  
    system = ClaspModel(cfg)  
    
    if cfg.resume_ckpt_path is not None:
        ckpt_path = cfg.resume_ckpt_path
        print(f"Training resumed from : {ckpt_path}")
    else:
        ckpt_path = None
  
    # Train the model  
    trainer.fit(model=system, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)  
  
  
if __name__ == '__main__':  
    main()  
