import torch.nn as nn
from transformers import AutoModel

class HuggingFaceEncoder(nn.Module):
    def __init__(self, cfg):
        super(HuggingFaceEncoder, self).__init__()

        self.text_encoder = AutoModel.from_pretrained(cfg.hf_textencoder_model_id)
        self.n_layers = cfg.textmlp_n_layers
        self.hidden_dim = cfg.textmlp_hidden_dim
        self.embedding_dim = cfg.embedding_dim
        self.dropout_prob = cfg.textmlp_dropout_prob

        # Freeze all parameters in the text encoder model
        if cfg.freeze_text_encoders:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # Get the output dimension. Handles models without a pooler_output
        self.encoder_output_dim = self.text_encoder.config.hidden_size

        # Create MLP layers with ReLU activations and BatchNorm except for the last layer, and add Dropout conditionally
        mlp_layers = []
        for i in range(self.n_layers):
            input_dim = self.encoder_output_dim if i == 0 else self.hidden_dim
            output_dim = self.embedding_dim if i == self.n_layers - 1 else self.hidden_dim
            mlp_layers.append(nn.Linear(input_dim, output_dim))
            if i < self.n_layers - 1:
                mlp_layers.append(nn.BatchNorm1d(output_dim))
                mlp_layers.append(nn.ReLU())
                if self.dropout_prob > 0:  # Only add dropout if dropout probability is greater than zero
                    mlp_layers.append(nn.Dropout(self.dropout_prob))

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, data):
        input_ids = data.tokenized_title["input_ids"]
        attention_mask = data.tokenized_title["attention_mask"]
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Handle models without a pooler_output (e.g., DeBERTa-v3), use the CLS token embedding
        if hasattr(outputs, 'pooler_output'):
             encoder_outputs = outputs.pooler_output
        else:
            encoder_outputs = outputs.last_hidden_state[:, 0, :]

        text_embeddings = self.mlp(encoder_outputs)

        return text_embeddings