import sys

# Add the parent directory to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from hydra import initialize, compose

import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.nn.functional import cosine_similarity
import pytorch_lightning as pl

from dataloaders.dataset import ClaspDataset
from dataloaders.common import seed_worker
from models.contrastive import ClaspModel
from utils.embedding_utils import encode_texts
from train_pretraining import load_metadata_and_embeddings

def calculate_material_category_similarities(target_embeddings, categories, tokenizer, text_encoder, cfg, device):
    """
    Calculate similarity between multiple material embeddings and category text embeddings.
    """
    category_embeddings = [encode_texts([category], tokenizer, text_encoder, cfg, device) for category in categories]
    category_embeddings = torch.stack(category_embeddings).squeeze(1)
    similarities = cosine_similarity(target_embeddings[:, None, :], category_embeddings[None, :, :], dim=2)
    return similarities

def filter_material_functions(titles_series, keywords, threshold=50):
    """
    Filter material function keywords that appear more than a threshold in the titles.
    """
    return [keyword for keyword in keywords if titles_series.str.contains(keyword, case=False).sum() >= threshold]
    
def calculate_roc_curve(dist_label_pairs):
    """
    Calculate ROC curve from a DataFrame of distance and label pairs.
    """
    cumsum_pos = 0
    roc_data = []
    total_num = len(dist_label_pairs)
    total_pos = dist_label_pairs["label"].sum()
    total_neg = total_num - total_pos
    for i in range(total_num):
        cumsum_pos += dist_label_pairs.iloc[i]['label']
        cumsum_neg = i + 1 - cumsum_pos
        roc_data.append([cumsum_neg / total_neg, cumsum_pos / total_pos])
    roc_data.sort()
    roc_data.insert(0, [0.0, 0.0])
    return roc_data

def calculate_average_precision(dist_label_pairs):
    """
    Calculate average precision from a DataFrame of distance and label pairs.
    """
    ap = average_precision_score(
        y_true=dist_label_pairs['label'],
        y_score=-dist_label_pairs['dist']  # Use negative distance as score
    )
    return ap

def downsample_false_labels(dist_label_pairs, random_state=42):
    """
    Downsample false labels to match the number of true labels.
    """
    true_labels = dist_label_pairs[dist_label_pairs['label'] == True]
    false_labels = dist_label_pairs[dist_label_pairs['label'] == False]
    if len(true_labels) >= len(false_labels):
        return dist_label_pairs
    downsampled_false_labels = false_labels.sample(n=len(true_labels), random_state=random_state)
    return pd.concat([true_labels, downsampled_false_labels]).sample(frac=1, random_state=42).reset_index(drop=True)

material_functions = [
 'ferromagnetic',
 'ferroelectric',
 'semiconductor',
 'electroluminescence',
 'thermoelectric',
 'superconductor']

keyword_variations = {
    'ferromagnetic': ['ferromagnetic', 'ferromagnetism'],
    'ferroelectric': ['ferroelectric', 'ferroelectricity'],
    'semiconductor': ['semiconductor', 'semiconductive', 'semiconductivity'],
    'electroluminescence': ['electroluminescence', 'electroluminescent'],
    'thermoelectric': ['thermoelectric', 'thermoelectricity'],
    'superconductor': ['superconductor', 'superconductive', 'superconductivity'],
}

def eval_roc(df, output_cry, tokenizer, text_encoder, cfg, keyword_variations):
    """
    Evaluate ROC curve for each keyword (no keyword variation handling).
    """
    plt.figure(figsize=(10, 7))
    auc_scores = []
    auc_scores_downsampled = []
    roc_data_list = []
    ap_list = []
    ap_list_downsampled = []
    data_size_list = []
    data_size_downsampled_list = []

    for query_keyword in tqdm(list(keyword_variations.keys())):
        category_embedding = encode_texts([query_keyword], tokenizer, text_encoder, cfg, device)
        dist = 1 - cosine_similarity(output_cry[:, None, :], category_embedding[None, :, :], dim=2)
        dist_label_pairs = pd.DataFrame({
            "label": df['titles'].str.contains(query_keyword, case=False),
            "dist": dist.squeeze()
        })
        dist_label_pairs = dist_label_pairs.sort_values("dist", ascending=True)
        dist_label_pairs_downsampled = downsample_false_labels(dist_label_pairs)
        roc_data = calculate_roc_curve(dist_label_pairs)
        roc_data_list.append(roc_data)
        roc_data_downsampled = calculate_roc_curve(dist_label_pairs_downsampled)
        x = [data[0] for data in roc_data]
        y = [data[1] for data in roc_data]
        plt.plot(x, y, label=query_keyword)
        auc_score = roc_auc_score(y_true=dist_label_pairs['label'], y_score=-dist_label_pairs['dist'])
        auc_scores.append(auc_score)
        auc_score_downsampled = roc_auc_score(y_true=dist_label_pairs_downsampled['label'], y_score=-dist_label_pairs_downsampled['dist'])
        auc_scores_downsampled.append(auc_score_downsampled)
        print(f'AUC for {query_keyword}: {auc_score:.4f}')
        ap = calculate_average_precision(dist_label_pairs)
        ap_list.append(ap)
        ap_downsampled = calculate_average_precision(dist_label_pairs_downsampled)
        ap_list_downsampled.append(ap_downsampled)
        data_size_list.append(len(dist_label_pairs))
        data_size_downsampled_list.append(len(dist_label_pairs_downsampled))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.grid(True)
    plt.legend()
    fig = plt.gcf()

    map = np.mean(ap_list)
    map_downsampled = np.mean(ap_list_downsampled)
    ap_df = pd.DataFrame({
        "keyword": list(keyword_variations.keys()),
        "average_precision": ap_list,
        'map': map,
        "average_precision_downsampled": ap_list_downsampled,
        'map_downsampled': map_downsampled,
        "data_size": data_size_list,
        "data_size_downsampled": data_size_downsampled_list
    })
    print(f"ap_df:\n{ap_df}")

    average_auc = np.mean(auc_scores)
    average_auc_downsampled = np.mean(auc_scores_downsampled)
    roc_auc_df = pd.DataFrame({
        "keyword": list(keyword_variations.keys()),
        "auc_score": auc_scores,
        'average_roc_auc': average_auc,
        "auc_score_downsampled": auc_scores_downsampled,
        'average_roc_auc_downsampled': average_auc_downsampled,
        "data_size": data_size_list,
        "data_size_downsampled": data_size_downsampled_list
    })
    print(f"roc_auc_df:\n{roc_auc_df}")
    return roc_auc_df, fig, roc_data_list, ap_df

def eval_roc_for_keyword_variations(df, output_cry, tokenizer, text_encoder, cfg, keyword_variations):
    """
    Evaluate ROC curve for each keyword, considering keyword variations.
    """
    print("eval_roc_for_keyword_variations called")
    plt.figure(figsize=(10, 7))
    auc_scores = []
    auc_scores_downsampled = []
    roc_data_list = []
    ap_list = []
    ap_list_downsampled = []
    data_size_list = []
    data_size_downsampled_list = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for query_keyword in tqdm(list(keyword_variations.keys())):
        variations = keyword_variations.get(query_keyword, [query_keyword])
        category_embeddings = [encode_texts([variation], tokenizer, text_encoder, cfg, device) for variation in variations]
        dists = [1 - cosine_similarity(output_cry[:, None, :], category_embedding[None, :, :], dim=2) for category_embedding in category_embeddings]
        min_dist = torch.min(torch.stack(dists), dim=0)[0]
        label = np.any([df['titles'].str.contains(variation, case=False) for variation in variations], axis=0)
        dist_label_pairs = pd.DataFrame({"label": label, "dist": min_dist.squeeze()})
        dist_label_pairs = dist_label_pairs.sort_values("dist", ascending=True)
        dist_label_pairs_downsampled = downsample_false_labels(dist_label_pairs)
        roc_data = calculate_roc_curve(dist_label_pairs)
        roc_data_list.append(roc_data)
        roc_data_downsampled = calculate_roc_curve(dist_label_pairs_downsampled)
        x = [data[0] for data in roc_data]
        y = [data[1] for data in roc_data]
        plt.plot(x, y, label=query_keyword)
        auc_score = roc_auc_score(y_true=dist_label_pairs['label'], y_score=-dist_label_pairs['dist'])
        auc_scores.append(auc_score)
        auc_score_downsampled = roc_auc_score(y_true=dist_label_pairs_downsampled['label'], y_score=-dist_label_pairs_downsampled['dist'])
        auc_scores_downsampled.append(auc_score_downsampled)
        print(f'AUC for {query_keyword}: {auc_score:.4f}')
        ap = calculate_average_precision(dist_label_pairs)
        ap_list.append(ap)
        ap_downsampled = calculate_average_precision(dist_label_pairs_downsampled)
        ap_list_downsampled.append(ap_downsampled)
        data_size_list.append(len(dist_label_pairs))
        data_size_downsampled_list.append(len(dist_label_pairs_downsampled))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.grid(True)
    plt.legend()
    fig = plt.gcf()

    map = np.mean(ap_list)
    map_downsampled = np.mean(ap_list_downsampled)
    ap_df = pd.DataFrame({
        "keyword": list(keyword_variations.keys()),
        "average_precision": ap_list,
        'map': map,
        "average_precision_downsampled": ap_list_downsampled,
        'map_downsampled': map_downsampled,
        "data_size": data_size_list,
        "data_size_downsampled": data_size_downsampled_list
    })
    print(f"ap_df:\n{ap_df}")

    average_auc = np.mean(auc_scores)
    average_auc_downsampled = np.mean(auc_scores_downsampled)
    roc_auc_df = pd.DataFrame({
        "keyword": list(keyword_variations.keys()),
        "auc_score": auc_scores,
        'average_roc_auc': average_auc,
        "auc_score_downsampled": auc_scores_downsampled,
        'average_roc_auc_downsampled': average_auc_downsampled,
        "data_size": data_size_list,
        "data_size_downsampled": data_size_downsampled_list
    })
    print(f"roc_auc_df:\n{roc_auc_df}")
    return roc_auc_df, fig, roc_data_list, ap_df




if __name__ == "__main__":
    target_dataset = "val"
    # target_dataset = "test"
    print(f"target dataset: {target_dataset}")
    print("loading metadata...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use environment variables or relative paths
    metadata_path = os.environ.get('EVAL_METADATA_PATH', 
                                 'data/cod_metadata_20240331_splitted_remaining.csv')
    cod_basepath = os.environ.get('COD_PATH', '/cod')
    metadata_and_embeddings = load_metadata_and_embeddings(load_path=metadata_path,
                                                       cod_basepath=cod_basepath)  
    
    # Use environment variable or relative path for dataset
    dataset_rootpath = os.environ.get('DATASET_ROOT_PATH', 'data/cod_full_20240331')
    print(f"dataset_rootpath: {dataset_rootpath}")

    dataset = ClaspDataset(input_dataframe=metadata_and_embeddings, 
                        tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased'), 
                        max_token_length=64,
                        root=dataset_rootpath)  

    # Prepare data loaders for train, validation and test  
    dataset_size = len(dataset)  
    train_size = int(0.8 * dataset_size)  
    val_size = int(0.1 * dataset_size)  
    test_size = dataset_size - train_size - val_size  

    generator = torch.Generator().manual_seed(42)  
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)  

    # train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, num_workers=2, drop_last=True,
    #                         pin_memory=True,persistent_workers=True,
    #                         worker_init_fn=seed_worker)  
    # val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False, num_workers=2, drop_last=False,
    #                         pin_memory=True,persistent_workers=True,
    #                         worker_init_fn=seed_worker)  
    # test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False, num_workers=2, drop_last=False,
    #                         pin_memory=True,persistent_workers=True,
    #                         worker_init_fn=seed_worker)  

    if target_dataset == "val":
        df = pd.DataFrame({"titles":[val_dataset.dataset.data["title"][i] for i in val_dataset.indices], 
                    "id": [val_dataset.dataset.data["material_id"][i] for i in val_dataset.indices]})
        dataloader = DataLoader(val_dataset, batch_size=2048, shuffle=False, num_workers=2, drop_last=False,
                            pin_memory=True,persistent_workers=True,
                            worker_init_fn=seed_worker) 
    elif target_dataset == "test":
        df = pd.DataFrame({"titles":[test_dataset.dataset.data["title"][i] for i in test_dataset.indices], 
                    "id": [test_dataset.dataset.data["material_id"][i] for i in test_dataset.indices]})
        dataloader = DataLoader(test_dataset, batch_size=2048, shuffle=False, num_workers=2, drop_last=False,
                            pin_memory=True,persistent_workers=True,
                            worker_init_fn=seed_worker) 

    # evaluation target checkpoint path
    config_paths = [
    # "../outputs/2024-05-24/clip_s2_lr2e-5_1k/22-14-06/version_0/",
    # "../outputs/2024-06-17/ft_clip_s2_lr1e-6_0616_ep2050/05-06-24/version_0/",
    # "../outputs/2024-10-03/clip_s2_lr2e-5/09-23-06/version_0/",
    # "../outputs/2024-10-14/ft_clip_s2_lr1e-6_1003_ep2050/04-50-01/version_0/",
    # "../outputs/2024-08-14/ft_cosface_s3_m03_lr1e-6_0812_ep2050/13-26-59/version_0",
    # "../outputs/2024-08-14/ft_cosface_s3_m05_lr1e-6_0813_ep2050/13-36-10/version_0",
    # "../outputs/2024-08-13/cosface_s3_m05_lr2e-5/08-01-05/version_0",
    # "../outputs/2024-08-12/cosface_s3_m03_lr2e-5/15-37-42/version_0",
    # "../outputs/2024-11-10/debertav3_cosface_s3_m05_lr2e-5_bs1536/12-41-43/version_0/",
    "../outputs/2025-01-16/ft_clip_s2_lr1e-6_1225_ep2050/17-21-13/version_0/",
    "../outputs/2024-12-25/clip_s2_lr2e-5/06-07-46/version_0/",
    # "../outputs/2025-01-16/ft_cosface_s3_m05_lr1e-6_1222_ep2050/17-12-01/version_0/",
    ]
    # Prepare dataloader and df before config_path loop
    if target_dataset == "val":
        df = pd.DataFrame({"titles": [val_dataset.dataset.data["title"][i] for i in val_dataset.indices],
                          "id": [val_dataset.dataset.data["material_id"][i] for i in val_dataset.indices]})
        dataloader = DataLoader(val_dataset, batch_size=2048, shuffle=False, num_workers=2, drop_last=False,
                                pin_memory=True, persistent_workers=True, worker_init_fn=seed_worker)
    elif target_dataset == "test":
        df = pd.DataFrame({"titles": [test_dataset.dataset.data["title"][i] for i in test_dataset.indices],
                          "id": [test_dataset.dataset.data["material_id"][i] for i in test_dataset.indices]})
        dataloader = DataLoader(test_dataset, batch_size=2048, shuffle=False, num_workers=2, drop_last=False,
                                pin_memory=True, persistent_workers=True, worker_init_fn=seed_worker)

    for config_path in tqdm(config_paths):
        print(f"config path: {config_path}")
        checkpoint_dir = f"{config_path}/model_checkpoint/"
        export_dir_root = os.environ.get('EVAL_RESULTS_DIR', 'eval_results')
        os.makedirs(export_dir_root, exist_ok=True)

        checkpoint_files = ["last.ckpt"]
        if any(keyword in config_path for keyword in ["ft_", "full_"]):
            checkpoint_files.extend(f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt'))

        for checkpoint_file in checkpoint_files:
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
            print(f"check point: {checkpoint_path}")
            with initialize(config_path, version_base='1.1'):
                cfg = compose(config_name="hparams")
                cfg.freeze_text_encoders = True

                model = ClaspModel.load_from_checkpoint(checkpoint_path, cfg=cfg, train_loader=None, val_loader=None)
                model.to(device)
                model.eval()

                tokenizer = AutoTokenizer.from_pretrained(cfg.hf_textencoder_model_id)
                text_encoder = model.model_text.to(device)
                from utils.embedding_utils import predict_embeddings
                output_cry, output_text = predict_embeddings(model, dataloader, device)
                save_filename_root = checkpoint_path.replace("../outputs/", "").replace("/", "_").replace("model_checkpoint", "")

                # without consider keyword variations
                auc_df, fig, roc_data, ap_df = eval_roc(df, output_cry, tokenizer, text_encoder, cfg, keyword_variations)
                plt.tight_layout()
                fig.savefig(f'{export_dir_root}/{target_dataset}/{save_filename_root}_roc_curve_{target_dataset}.pdf')
                plt.close()
                auc_df.to_csv(f'{export_dir_root}/{target_dataset}/{save_filename_root}_roc_table_{target_dataset}.csv', index=False)

                with open(f'{export_dir_root}/{target_dataset}/{save_filename_root}_roc_data_{target_dataset}.pkl', 'wb') as f:
                    pickle.dump(roc_data, f)
                ap_df.to_csv(f'{export_dir_root}/{target_dataset}/{save_filename_root}_ap_data_{target_dataset}.csv', index=False)

                # consider keyword variations
                auc_df, fig, roc_data, ap_df = eval_roc_for_keyword_variations(df, output_cry, tokenizer, text_encoder, cfg, keyword_variations)
                plt.tight_layout()
                fig.savefig(f'{export_dir_root}/{target_dataset}/{save_filename_root}_roc_curve_keyword_variations_{target_dataset}.pdf')
                plt.close()
                auc_df.to_csv(f'{export_dir_root}/{target_dataset}/{save_filename_root}_roc_table_keyword_variations_{target_dataset}.csv', index=False)

                with open(f'{export_dir_root}/{target_dataset}/{save_filename_root}_roc_data_keyword_variations_{target_dataset}.pkl', 'wb') as f:
                    pickle.dump(roc_data, f)
                ap_df.to_csv(f'{export_dir_root}/{target_dataset}/{save_filename_root}_ap_data_keyword_variations_{target_dataset}.csv', index=False)


# Example:
# python eval_zero_shot_roc.py 