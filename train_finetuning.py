#!/usr/bin/env python  
# coding: utf-8  
  
import sys  
import os
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd  
import torch  
from tqdm import tqdm
from torch.utils.data import random_split, Subset
from torch_geometric.loader import DataLoader  
from pytorch_lightning import Trainer, loggers  
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
from omegaconf import DictConfig, OmegaConf  
from transformers import AutoTokenizer
from dataloaders.dataset import ClaspDataset, ClaspOnDiskDataset
from dataloaders.common import generate_full_path, seed_worker  
from models.contrastive import ClaspModel  
from utils.finetune_utils import create_dataframe_from_json_strings, remove_empty_entries, exclude_keywords
  
def load_metadata_and_embeddings(load_path, cod_basepath="/cif"):  
    """Load the embedding metadata."""  
    metadata_df = pd.read_csv(load_path)  
    metadata_df = metadata_df.dropna(subset=['title'])
    metadata_df['cif_path'] = metadata_df['file'].astype(str).apply(lambda x: generate_full_path(x, base_path=cod_basepath))
    return metadata_df

def load_caption_dataframe(json_path):
    keywords = pd.read_json(json_path)
    keywords_df = create_dataframe_from_json_strings(keywords["output_0"])
    keywords_to_exclude = ['Crystal Structure', 'X-ray diffraction', 'Neutron Diffraction', 'Powder Diffraction', "Single-Crystal X-ray Diffraction"]
    keywords_df = exclude_keywords(keywords_df, 'Keywords', keywords_to_exclude)

    keywords_df = remove_empty_entries(keywords_df, 'Keywords')
    return keywords_df
    

def prepare_datasets(cfg, tokenizer, metadata):
    if cfg.dataset_load_in_memory:
        dataset = ClaspDataset(input_dataframe=metadata, 
                            tokenizer=tokenizer, 
                            max_token_length=cfg.max_token_length,
                            root=cfg.dataset_cache_dir)  
    elif not cfg.dataset_load_in_memory:
        dataset = ClaspOnDiskDataset(input_dataframe=metadata, 
                        tokenizer=tokenizer, 
                        max_token_length=cfg.max_token_length,
                        root=cfg.dataset_cache_dir)  
    else:
        raise ValueError
    
    # Split dataset into train, validation and test  
    dataset_size = len(dataset)  
    train_size = int(0.8 * dataset_size)  
    val_size = int(0.1 * dataset_size)  
    test_size = dataset_size - train_size - val_size  
  
    seed = 42  
    generator = torch.Generator().manual_seed(seed)  
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)  

    # Prepare dataset for finetuning
    material_id_train = [int(dataset[i]["material_id"]) for i in train_dataset.indices]
    material_id_val = [int(dataset[i]["material_id"]) for i in val_dataset.indices]
    material_id_test = [int(dataset[i]["material_id"]) for i in test_dataset.indices]

    keywords_df = load_caption_dataframe(json_path=cfg.finetuning_caption_json_path)

    metadata['file'] = metadata['file'].astype(int)
    
    # IDが文字列型の場合があるので、intに変換する。変換できない場合はNoneにしてからdrop
    def safe_convert_to_int(value):
        try:
            return int(value)
        except ValueError:
            return None
        except TypeError:
            return None

    keywords_df['ID'] = keywords_df['ID'].apply(safe_convert_to_int)
    keywords_df = keywords_df.dropna(subset=['ID'])
    keywords_df['ID'] = keywords_df['ID'].astype(int)

    merged_df = pd.merge(metadata, keywords_df, left_on='file', right_on='ID')
  
    merged_df.rename(columns={"title":"title_original"}, inplace=True)

    merged_df.drop(columns="ID", inplace=True)
    # merged_df.drop(columns="Title", inplace=True)
    merged_df.rename(columns={"Keywords":"title"}, inplace=True)  

    def list_to_string(lst):
        return ', '.join(lst)

    assert len(merged_df) > 1, "Error: merged_df is empty. Please check the input metadata and keywords data."

    merged_df['title'] = merged_df['title'].apply(list_to_string)

    train_df = merged_df[merged_df['file'].isin(material_id_train)]
    val_df = merged_df[merged_df['file'].isin(material_id_val)]
    test_df = merged_df[merged_df['file'].isin(material_id_test)]

    print("loading val dataset...")
    val_dataset_ft = ClaspDataset(input_dataframe=val_df, 
                                tokenizer=tokenizer, 
                                max_token_length=cfg.max_token_length,
                                root=os.path.join(cfg.finetuning_dataset_cache_dir, "_val"))
    print("loading test dataset...")
    test_dataset_ft = ClaspDataset(input_dataframe=test_df, 
                                tokenizer=tokenizer, 
                                max_token_length=cfg.max_token_length,
                                root=os.path.join(cfg.finetuning_dataset_cache_dir, "_test"))  
    print("loading train dataset...")
    train_dataset_ft = ClaspDataset(input_dataframe=train_df, 
                                tokenizer=tokenizer, 
                                max_token_length=cfg.max_token_length,
                                root=os.path.join(cfg.finetuning_dataset_cache_dir, "_train"))
    return train_dataset_ft, val_dataset_ft, test_dataset_ft
  

def lazy_prepare_datasets(cfg):
    """
    あらかじめ`prepare_datasets`を実行してデータセットの前処理を行い、データセットが書き出されている場合はこの関数でロードすると速い
    """

    datasets = {}
    for suffix in ['_train', '_val', '_test']:
        dataset_dir = os.path.join(cfg.finetuning_dataset_cache_dir, suffix)
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(f"The {suffix[1:]} dataset has not been exported at {dataset_dir}")
        print(f"loading {suffix[1:]} dataset...")
        datasets[suffix] = ClaspDataset(input_dataframe=None,
                                        tokenizer=None,
                                        max_token_length=cfg.max_token_length,
                                        root=dataset_dir)
                                        
    return datasets['_train'], datasets['_val'], datasets['_test']


def filter_and_replace_title(dataset, keywords_df):
    """
    データセットをキーワードでフィルタリングし、title属性を置換する

    Args:
        dataset (Dataset): データセット
        keywords_df (DataFrame): キーワードのデータフレーム

    Returns:
        Dataset: フィルタリングされ、title属性が置換されたデータセット
    """
    filtered_indices = []
    for i in tqdm(range(len(dataset))):
        if dataset[i]["material_id"] in keywords_df["ID"].values:
            dataset[i]["title"] = ', '.join(keywords_df[keywords_df["ID"] == dataset[i]["material_id"]]["Keywords"].values[0])
            filtered_indices.append(i)
    return Subset(dataset, filtered_indices)


def setup_trainer(cfg):  
    """Setup the PyTorch Lightning trainer."""  
    print(os.getcwd())
    logger = loggers.TensorBoardLogger(os.getcwd(), name=None)  
    logger.log_hyperparams(cfg)
  
    checkpoint_callback = ModelCheckpoint(save_top_k=cfg.model_checkpoint_save_top_k,  
                                          monitor=None, save_last=True,  
                                          dirpath=logger.log_dir+'/model_checkpoint', every_n_epochs=10)
  
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
                      check_val_every_n_epoch=cfg.check_val_every_n_epoch,
                      gradient_clip_val=2.0,
                      gradient_clip_algorithm="value")  
    return trainer  
  

@hydra.main(version_base=None, config_path="./configs", config_name="soroban_finetuning")
def main(cfg : DictConfig):  
    print(OmegaConf.to_yaml(cfg))  
    # print("cwd: ",os.getcwd()) # logdir
    # Ignore warnings  
    # warnings.filterwarnings("ignore")  
    is_1st_run = True

    if is_1st_run:  
        # Load embedding metadata  
        metadata = load_metadata_and_embeddings(load_path=cfg.input_pickle_path,
                                                            cod_basepath=cfg.cod_basepath) 
        # Create the dataset  
        tokenizer = AutoTokenizer.from_pretrained(cfg.hf_textencoder_model_id)
        train_dataset, val_dataset, test_dataset = prepare_datasets(cfg=cfg, tokenizer=tokenizer, metadata=metadata)
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        del(metadata)
        del(tokenizer)

    else:   
        train_dataset, val_dataset, test_dataset = lazy_prepare_datasets(cfg=cfg)
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=8, drop_last=True,
                          pin_memory=True,persistent_workers=True,
                          worker_init_fn=seed_worker)  
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=8, drop_last=False,
                            pin_memory=True,persistent_workers=True,
                            worker_init_fn=seed_worker)  
    print("dataset loaded")
    trainer = setup_trainer(cfg)  
    system = ClaspModel(cfg)  
    
    if cfg.resume_ckpt_path is not None:
        ckpt_path = cfg.resume_ckpt_path
        print(f"Training resumed from : {ckpt_path}")
    else:
        ckpt_path = None
  
    # Train the model  
    # todo: BERTの全レイヤーのunfreeze
    trainer.fit(model=system, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)  
  
  
if __name__ == '__main__':  
    main()  
