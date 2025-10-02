import warnings    
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def clip_loss(text_embedding, crystal_embedding, params):
    """
    Calculate cross entropy loss for text and crystal embeddings.
    This implementation is based on `https://github.com/moein-shariatnia/OpenAI-CLIP/blob/master/CLIP.py` .

    Args:
      text_embedding: Tensor, shape [batch_size, embedding_size], feature embeddings of texts.
      crystal_embedding: Tensor, shape [batch_size, embedding_size], feature embeddings of crystal structures.
      params: Namespace or dict, parameters for loss calculation (loss_scale, reduction_mode).

    Returns:
      loss: float tensor, the calculated triplet loss.

    """
    if hasattr(params, "loss_scale"):
        loss_scale = params.loss_scale
    else:
        loss_scale = 1.0
    logits_per_crystal = loss_scale * crystal_embedding @ text_embedding.T
    logits_per_text = loss_scale * text_embedding @ crystal_embedding.T

    labels = torch.arange(logits_per_text.shape[0], dtype=torch.long, device=logits_per_text.device)

    total_loss = (
        F.cross_entropy(logits_per_crystal, labels) +
        F.cross_entropy(logits_per_text, labels)
    ) / 2.

    return total_loss

def cosface_loss(text_embedding, crystal_embedding, params):
    """
    Calculate CosFace loss for text and crystal embeddings.

    Args:
      text_embedding: Tensor, shape [batch_size, embedding_size], feature embeddings of texts.
      crystal_embedding: Tensor, shape [batch_size, embedding_size], feature embeddings of crystals.
      m: float, margin

    Returns:
      loss: float tensor, the calculated CosFace loss.
    """
    margin = params.margin
    if hasattr(params, "loss_scale"):
        loss_scale = params.loss_scale
    else:
        loss_scale = 10.0

    # Compute the cosine similarity matrix
    cosine_matrix = text_embedding @ crystal_embedding.T
    
    # Get the number of samples
    batch_size = text_embedding.size(0)

    # Create the label vector
    labels = torch.arange(batch_size, dtype=torch.long, device=text_embedding.device)

    # Apply the margin to the diagonal (where labels match)
    margin_matrix = torch.eye(batch_size, device=text_embedding.device) * margin
    cosine_with_margin = (cosine_matrix - margin_matrix)*loss_scale

    # Calculate the loss using cross-entropy
    loss_text = F.cross_entropy(cosine_with_margin, labels)
    loss_crystal = F.cross_entropy(cosine_with_margin.T, labels)

    # Combine losses
    total_loss = (loss_text + loss_crystal) / 2

    return total_loss


def arcface_loss(text_embedding, crystal_embedding, params):
    """
    Calculate ArcFace loss for text and crystal embeddings.
    
    Args:
        text_embedding: Tensor, shape [batch_size, embedding_size], feature embeddings of texts.
        crystal_embedding: Tensor, shape [batch_size, embedding_size], feature embeddings of crystals.
        params: Object containing the following attributes:
            margin: float, angular margin
            scale: float, scaling factor for input embeddings
    
    Returns:
        loss: float tensor, the calculated ArcFace loss.
    """
    margin = params.margin
    if hasattr(params, "loss_scale"):
        loss_scale = params.loss_scale
    else:
        loss_scale = 10.0

    # Compute the cosine similarity matrix
    cosine_matrix = text_embedding @ crystal_embedding.T

    # Get the number of samples
    batch_size = text_embedding.size(0)

    # Create the label vector
    labels = torch.arange(batch_size, dtype=torch.long, device=text_embedding.device)

    # Apply the margin to the diagonal (where labels match)
    diag_cos = cosine_matrix.diagonal()
    eps = 1e-5
    diag_cos = torch.clamp(diag_cos, -1+eps, 1-eps)
    diag_theta_margin = torch.arccos(diag_cos) + margin # put margin in angle space
    diag_cos_margin = torch.cos(diag_theta_margin) # invert to cos

    # Calculate the loss using cross-entropy
    cosine_matrix_with_margin = torch.diagonal_scatter(cosine_matrix, diag_cos_margin, 0)
    loss_text = F.cross_entropy(loss_scale * (cosine_matrix + cosine_matrix_with_margin), labels)
    loss_crystal = F.cross_entropy(loss_scale * (cosine_matrix.T + cosine_matrix_with_margin.T), labels)

    # Combine losses
    total_loss = (loss_text + loss_crystal) / 2

    return total_loss


class SigLIPLoss(nn.Module):
    """
    PyTorch Module implementation for SigLIP Loss.
    Assumes input embeddings are already L2 normalized.
    This module holds the learnable temperature and bias parameters.

    Args:
        params (Namespace or dict): Parameters for loss initialization. Expects:
            - loss_scale (float, optional): Initial value for the logit scale (log temperature). Defaults to log(10.0).
            - initial_logit_bias (float, optional): Initial value for the logit bias. Defaults to -10.0.
    """
    def __init__(self, params):
        super().__init__()
        # Get initial values from params, use defaults if not present
        # Use getattr to support both dict and Namespace types
        initial_scale = getattr(params, "loss_scale", torch.log(torch.tensor(10.0)))
        initial_bias = getattr(params, "initial_logit_bias", -10.0)

        if isinstance(initial_scale, torch.Tensor):
             initial_scale = initial_scale.item()  # Ensure float if passed as tensor
        if isinstance(initial_bias, torch.Tensor):
             initial_bias = initial_bias.item()  # Ensure float if passed as tensor

        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.tensor(initial_scale, dtype=torch.float32))
        # Learnable bias parameter
        self.logit_bias = nn.Parameter(torch.tensor(initial_bias, dtype=torch.float32))

        print(f"Initialized SigLIP Loss Module with logit_scale={self.logit_scale.item():.4f} (t={torch.exp(self.logit_scale).item():.4f}) and logit_bias={self.logit_bias.item():.4f}")

    def forward(self, text_embedding: torch.Tensor, crystal_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute the SigLIP loss using the internal learnable parameters.

        Args:
            text_embedding (torch.Tensor): L2 normalized embeddings for text, shape (batch_size, embedding_dim).
            crystal_embedding (torch.Tensor): L2 normalized embeddings for crystals (or images), shape (batch_size, embedding_dim).

        Returns:
            torch.Tensor: The computed SigLIP loss (scalar).
        """
        batch_size = text_embedding.shape[0]
        device = text_embedding.device

        # Assume input is already L2 normalized
        text_embedding_norm = text_embedding
        crystal_embedding_norm = crystal_embedding

        # Get temperature and bias
        # Note: Clamping logit_scale is not mentioned in the paper, but may help stability if needed.
        # Example: t = torch.exp(self.logit_scale.clamp(max=4.605)) # max=log(100)
        t = torch.exp(self.logit_scale)
        b = self.logit_bias

        # Compute pairwise similarity logits
        # (batch_size, embedding_dim) @ (embedding_dim, batch_size) -> (batch_size, batch_size)
        # Rows: crystal_embedding, Columns: text_embedding
        logits = crystal_embedding_norm @ text_embedding_norm.t() * t + b
        self.logits = logits

        # Create label matrix (+1: positive pair, -1: negative pair)
        labels = 2 * torch.eye(batch_size, device=device, dtype=logits.dtype) - torch.ones(batch_size, batch_size, device=device, dtype=logits.dtype)

        # Compute pairwise sigmoid loss
        # loss_ij = -log(sigmoid(label_ij * logit_ij))
        loss_pairwise = -F.logsigmoid(labels * logits)

        # Sum loss over all pairs and normalize by batch size
        loss = torch.sum(loss_pairwise) / batch_size

        return loss
