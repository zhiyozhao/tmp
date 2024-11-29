import torch
import torch.nn as nn
from transformers import AutoModel


class TaskSpecificVQA(nn.Module):
    def __init__(
        self,
        image_model_name="google/vit-base-patch16-224-in21k",
        text_model_name="bert-base-uncased",
        num_classes=1000,
        fusion_dim=512,
        dropout_prob=0.3,
    ):
        """
        Finetune task-specific pretrained image and text encoders for VQA.

        Args:
            image_model_name (str): Pretrained image encoder model name (e.g., ViT).
            text_model_name (str): Pretrained text encoder model name (e.g., BERT).
            num_classes (int): Number of answer classes for VQA.
            fusion_dim (int): Dimensionality of the hidden layer after concatenation.
            dropout_prob (float): Dropout probability for regularization.
        """
        super(TaskSpecificVQA, self).__init__()

        # Load pretrained image model using AutoModel
        self.image_encoder = AutoModel.from_pretrained(image_model_name)

        # Load pretrained text model using AutoModel
        self.text_encoder = AutoModel.from_pretrained(text_model_name)

        # Define a ReLU activation as a module attribute
        self.relu = nn.ReLU()

        # Define a deeper MLP head with ReLU activations and Dropout
        self.fc1 = nn.Linear(
            self.image_encoder.config.hidden_size
            + self.text_encoder.config.hidden_size,
            fusion_dim,
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(
            fusion_dim, fusion_dim
        )  # Adding an extra layer for more expressiveness
        self.fc3 = nn.Linear(
            fusion_dim, num_classes
        )  # Final output layer for classification

    def forward(self, pixel_values, input_ids, attention_mask):
        """
        Forward pass for VQA.

        Args:
            pixel_values (torch.Tensor): Image tensor (batch_size, 3, H, W).
            input_ids (torch.Tensor): Tokenized question input (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention mask for question (batch_size, seq_len).

        Returns:
            torch.Tensor: Logits (batch_size, num_classes).
        """
        # Get image and text embeddings from their respective encoders with return_dict=True
        image_outputs = self.image_encoder(
            pixel_values, return_dict=True
        )  # Using return_dict=True
        text_outputs = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )  # Using return_dict=True

        # Extract embeddings (usually from [CLS] token for BERT, or pooled features for ViT)
        image_embeds = image_outputs.pooler_output  # Accessing the pooled output (ViT)
        text_embeds = (
            text_outputs.pooler_output
        )  # Accessing the pooled output ([CLS] token for BERT)

        # Concatenate the image and text embeddings
        combined_features = torch.cat(
            [image_embeds, text_embeds], dim=1
        )  # (batch_size, combined_dim)

        # Pass through the MLP head with dropout and ReLU activations
        x = self.relu(self.fc1(combined_features))  # (batch_size, fusion_dim)
        x = self.dropout(x)  # Apply dropout for regularization
        x = self.relu(self.fc2(x))  # (batch_size, fusion_dim)
        logits = self.fc3(x)  # (batch_size, num_classes)

        return logits
