from typing import List, Tuple, Optional, Union, Dict, Any
import torch
from torch import Tensor
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

class TokenizedInput:
    """Container for tokenized text inputs."""
    
    def __init__(self, tokenized_title: Dict[str, Tensor]):
        """Initialize with tokenized title data.
        
        Args:
            tokenized_title: Dictionary containing 'input_ids' and 'attention_mask' tensors.
        """
        self.tokenized_title = tokenized_title

def predict_embeddings(
    model: torch.nn.Module, 
    data_loader: DataLoader, 
    device: torch.device
) -> Tuple[Tensor, Tensor]:
    """
    Extract embeddings from the model for all batches in the data loader.
    
    Args:
        model: The trained model to use for inference.
        data_loader: DataLoader containing the data to process.
        device: Device to run the model on (cuda/cpu).
        
    Returns:
        Tuple[Tensor, Tensor]: Crystal and text embeddings for all samples.
        
    Raises:
        RuntimeError: If model inference fails.
    """
    model.eval()  # Ensure model is in eval mode
    
    print("Predicting embeddings...")
    output_cry_list = []
    output_text_list = []
    
    try:
        with torch.no_grad():
            for batch in tqdm(data_loader):
                batch = batch.to(device)
                output_cry_run, output_text_run = model(batch)
                output_cry_list.append(output_cry_run.cpu())
                output_text_list.append(output_text_run.cpu())
    except Exception as e:
        raise RuntimeError(f"Failed to extract embeddings: {str(e)}")

    if not output_cry_list:
        raise RuntimeError("No embeddings were extracted. Data loader might be empty.")
        
    output_cry = torch.cat(output_cry_list, dim=0)
    output_text = torch.cat(output_text_list, dim=0)
    print("Embedding prediction completed.")
    return output_cry, output_text


def normalize_embedding(embedding: Tensor, norm_type: str) -> Tensor:
    """
    Normalize embeddings using the specified normalization method.
    
    Args:
        embedding: The embedding tensor to normalize (shape: [batch_size, embedding_dim]).
        norm_type: Type of normalization ('l2' or 'minmax').
        
    Returns:
        Tensor: Normalized embeddings.
        
    Raises:
        ValueError: If unsupported normalization type is provided.
    """
    if norm_type == 'l2':
        embedding /= embedding.norm(p=2, dim=1, keepdim=True)
    elif norm_type == 'minmax':
        embedding = (embedding - embedding.min(dim=1, keepdim=True)[0]) / (embedding.max(dim=1, keepdim=True)[0] - embedding.min(dim=1, keepdim=True)[0])
    else:
        raise ValueError(f"Unsupported normalization type: {norm_type}")
    return embedding


def encode_texts(
    text_list: List[str], 
    tokenizer: Any, 
    text_encoder: torch.nn.Module, 
    cfg: Any, 
    device: torch.device
) -> Tensor:
    """
    Encode a list of texts into embeddings.
    
    Args:
        text_list: List of text strings to encode.
        tokenizer: Tokenizer for processing text.
        text_encoder: Text encoder model.
        cfg: Configuration object containing embedding settings.
        device: Device to run the encoder on.
        
    Returns:
        Tensor: Text embeddings for all input texts.
        
    Raises:
        ValueError: If text_list is empty.
        RuntimeError: If encoding fails.
    """
    if not text_list:
        raise ValueError("text_list cannot be empty")
    
    text_encoder.eval()  # Ensure encoder is in eval mode
    
    try:
        encoded_input = tokenizer(text_list, padding=True, truncation=True, return_tensors='pt')
        
        data = TokenizedInput({
            "input_ids": encoded_input['input_ids'].to(device),
            "attention_mask": encoded_input['attention_mask'].to(device)
        })
        
        with torch.no_grad():
            embedding = text_encoder(data).cpu()
            
        if hasattr(cfg, 'embedding_normalize') and cfg.embedding_normalize is not None:
            embedding = normalize_embedding(embedding, cfg.embedding_normalize)
    except Exception as e:
        raise RuntimeError(f"Failed to encode texts: {str(e)}")
        
    return embedding


def calculate_material_category_similarities(
    target_embeddings: Tensor, 
    categories: List[str], 
    tokenizer: Any, 
    text_encoder: torch.nn.Module, 
    cfg: Any, 
    device: torch.device
) -> Tensor:
    """
    Calculate cosine similarities between material embeddings and category text embeddings.
    
    Args:
        target_embeddings: Material embeddings (shape: [num_materials, embedding_dim]).
        categories: List of category names.
        tokenizer: Tokenizer for processing category names.
        text_encoder: Text encoder model.
        cfg: Configuration object containing embedding settings.
        device: Device to run computations on.
        
    Returns:
        Tensor: Similarity scores (shape: [num_materials, num_categories]).
    """
    # Encode all categories at once for better efficiency
    category_embeddings = encode_texts(categories, tokenizer, text_encoder, cfg, device)

    all_similarities = cosine_similarity(target_embeddings[:, None, :], category_embeddings[None, :, :], dim=2)
    return all_similarities