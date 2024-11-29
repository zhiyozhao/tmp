import torch
import torch.nn as nn
from transformers import CLIPModel


class CLIPForVQA(nn.Module):
    def __init__(
        self,
        clip_name="openai/clip-vit-base-patch32",
        num_classes=1000,
        fusion_dim=512,
        dropout_prob=0.3,
    ):
        """
        Finetune CLIP for VQA with a deeper MLP head.

        Args:
            clip_name (str): Pretrained CLIP model name.
            num_classes (int): Number of answer classes for VQA.
            fusion_dim (int): Dimensionality of the hidden layer after concatenation.
            dropout_prob (float): Dropout probability for regularization.
        """
        super(CLIPForVQA, self).__init__()

        # Load the pretrained CLIP model
        self.clip = CLIPModel.from_pretrained(clip_name)

        # Define a ReLU activation as a module attribute
        self.relu = nn.ReLU()

        # Define a deeper MLP head with ReLU activations and Dropout
        self.fc1 = nn.Linear(self.clip.config.projection_dim * 2, fusion_dim)
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
        # Get image and text embeddings from CLIP
        clip_outputs = self.clip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Extract image and text embeddings
        image_embeds = clip_outputs.image_embeds  # (batch_size, image_embedding_dim)
        text_embeds = clip_outputs.text_embeds  # (batch_size, text_embedding_dim)

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
