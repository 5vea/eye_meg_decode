"""
This code is for the transformer model
It is really similar to our transformer classifier. Besides, it does not have data augmentation and the head is different.
Still a combined cls and GAP approach is used to get the final embedding.
A non linear layer is then used to project to the latent space.
Also another linear layer is trained for the vision encoder.

The image encoder is a pretrained ViT model from torchvision, but its parameters are not frozen. It will have a lower learning rate than the rest of the model.
See training script for more details.
"""

import numpy as np
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# positional encoding from https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch
# set dropout to zero, because it is no learnable embedding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 1700):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
            # this sucks, but we have to first optimize for this shape
        """
        x = x + self.pe[:x.size(0)]
        return x

class SelfAttention(nn.Module):
    """One head of self-attention: calculates attention for each token in relation to others."""

    def __init__(self, head_size, embedding_dim, dropout_rate):
        super().__init__()
        # Linear transformations for computing the key, query, and value matrices
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)

        """
        # Create a lower-triangular mask for future tokens (causal mask)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))"""
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass for self-attention head.

        Args:
        x (Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
        Tensor: Output tensor of shape (batch_size, sequence_length, head_size).
        """
        batch_size, sequence_length, embedding_dim = x.shape

        # Calculate key, query, and value matrices
        keys = self.key(x)  # Shape: (batch_size, sequence_length, head_size)
        queries = self.query(x)  # Shape: (batch_size, sequence_length, head_size)

        # Compute attention scores by taking dot product of queries and keys
        # Scaled by square root of head_size to maintain stable gradients
        attention_scores = queries @ keys.transpose(-2, -1) * (embedding_dim ** -0.5)

        # Convert attention scores to probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Calculate weighted sum of values
        values = self.value(x)
        output = attention_probs @ values  # Shape: (batch_size, sequence_length, head_size)

        return output


class MultiHeadAttention(nn.Module):
    """Combines multiple self-attention heads in parallel."""

    def __init__(self, num_heads, head_size, embedding_dim, dropout_rate):
        super().__init__()
        # Initialise multiple self-attention heads
        self.heads = nn.ModuleList([SelfAttention(head_size, embedding_dim, dropout_rate) for _ in range(num_heads)])
        # Project concatenated output of all heads back to embedding dimension
        self.proj = nn.Linear(head_size * num_heads, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass for multi-head attention.

        Args:
        x (Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
        Tensor: Output tensor of shape (batch_size, sequence_length, embedding_dim).
        """
        # Concatenate outputs from each head along the last dimension
        multi_head_output = torch.cat([head(x) for head in self.heads], dim=-1)

        # Apply final linear projection and dropout
        output = self.dropout(self.proj(multi_head_output))
        return output


class MLP(nn.Module):
    """Defines a feedforward neural network (MLP) for additional processing after attention."""

    def __init__(self, embedding_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),  # Expand the embedding dimension
            nn.GELU(), # swapped to GeLU
            nn.Linear(4 * embedding_dim, embedding_dim),  # Project back down
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        """
        Forward pass for the MLP.

        Args:
        x (Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
        Tensor: Processed tensor of the same shape.
        """
        return self.net(x)


class TransformerBlock(nn.Module):
    """Defines a single transformer block with self-attention and MLP layers."""

    def __init__(self, embedding_dim, num_heads, dropout_rate):
        super().__init__()
        head_size = embedding_dim // num_heads
        self.attention = MultiHeadAttention(num_heads, head_size, embedding_dim, dropout_rate)
        self.feedforward = MLP(embedding_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        """
        Forward pass for the transformer block.

        Args:
        x (Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
        Tensor: Output tensor of the same shape.
        """
        # Apply multi-head attention followed by layer normalisation
        x = x + self.attention(self.norm1(x))
        # Apply MLP followed by layer normalisation
        x = x + self.feedforward(self.norm2(x))
        return x

# needs to be adapted
class EyeTransformerEnc(nn.Module):
    """ A transformer encoder model for eye data classification. """

    def __init__(self, embedding_dim=64, num_heads=12, num_layers=1, dropout_rate=0):
        """
        Initialises the model with specified vocabulary size, embedding dimension, number of heads,
        number of transformer layers, and dropout rate.

        Args:
        embedding_dim (int): Dimension of the token embeddings.
        num_heads (int): Number of attention heads in the multi-head attention layers.
        num_layers (int): Number of transformer blocks to stack.
        dropout_rate (float): Dropout rate to apply within the model.
        n_classes (int): Number of classes for classification.
        """
        super().__init__()

        # exclusive convolution to project to embedding space
        # here we are just moving in time: 3 channels = x,y and pupil dilation
        self.conv_projection = nn.Conv1d(in_channels=3, out_channels=embedding_dim, kernel_size=1, stride=1)

        self.embedding_dim = embedding_dim

        # positional encoding: sine and cosine functions
        self.position_embedding = PositionalEncoding(embedding_dim)

        # Stack of transformer blocks
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(embedding_dim, num_heads=num_heads, dropout_rate=dropout_rate) for _ in
              range(num_layers)]
        )

        # Final layer normalisation for stable outputs
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Initialise weights for stability
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialises weights for linear and conv1d embedding layers."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Conv1d):
            nn.init.xavier_uniform_(module.weight)

    def forward(self, input_ids):
        """
        Forward pass of the model.

        Args:
        input_ids (Tensor): Tensor of eye signal --> will be tokenized and embedded with conv_projection

        Returns:
        Tensor: Predictions for classification.
        """
        batch_size, channels, sequence_length = input_ids.shape

        # cls token for classification
        cls_token = torch.zeros(batch_size, 1, self.embedding_dim).to(device)

        # Create token embeddings
        token_embeddings = self.conv_projection(input_ids)  # Shape: (batch_size, embedding_dim, sequence_length)
        # warning: dim needs to be swapped

        # this is annoying, but embedding dim and channel out need to be switched
        token_embeddings = torch.swapaxes(token_embeddings, 1,2) # Shape: (batch_size, sequence_length, embedding_dim)

        # concat cls token to embeddings
        token_embeddings = torch.cat((cls_token, token_embeddings), dim=1)  # Shape: (batch_size, sequence_length + 1, embedding_dim)

        # now it is on normal grounds, lets make it ugly for the function
        token_embeddings = torch.swapaxes(token_embeddings, 0, 1) # Shape: (sequence_length + 1, batch_size, embedding_dim)

        # apply positional encoding
        x = self.position_embedding(token_embeddings)

        # switch back to normal axes
        x = torch.swapaxes(x, 0, 1) # Shape: (batch_size, sequence_length + 1, embedding_dim)

        # Pass through stacked transformer blocks
        x = self.transformer_blocks(x)  # Shape: (batch_size, sequence_length + 1, embedding_dim)

        # Apply final layer normalisation
        x = self.layer_norm(x)  # Shape: (batch_size, sequence_length + 1, embedding_dim)

        # get embedded cls token
        cls_token = x[:, 0, :]  # Shape: (batch_size, embedding_dim)

        x = x[:, 1:, :]  # Shape: (batch_size, sequence_length, embedding_dim)

        # gap for sequence embeddings
        x = torch.mean(x, dim=1)  # Shape: (batch_size, embedding_dim)

        # norm
        x = self.layer_norm(x) # Shape: (batch_size, embedding_dim)

        # concat x and cls token
        x = torch.cat((x, cls_token), dim=1)  # Shape: (batch_size, 2 * embedding_dim)

        # flatten after batch size for final embedding
        x = torch.flatten(x, start_dim=1)

        return x

# A projection head to map encoder outputs to a shared embedding space
class ProjectionHead(nn.Module):
    """
    Copied from https://deeplearning-jupyterbook.github.io/notebooks/clip.html
    """
    def __init__(self, embedding_dim, projection_dim):
        super().__init__()
        # Linear layer to project the encoder's output to a lower-dimensional space
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()  # Activation function
        self.fc = nn.Linear(projection_dim, projection_dim)  # Fully connected layer
        self.layer_norm = nn.LayerNorm(projection_dim)  # Normalize embeddings

    def forward(self, x):
        # Project the input embeddings
        projected = self.projection(x)
        # Apply activation and a second linear layer
        x = self.gelu(projected)
        x = self.fc(x)
        # Add skip connection and normalize
        x = x + projected
        x = self.layer_norm(x)
        return x

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = torchvision.models.vit_b_32(weights=torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1)
        #self.model = nn.Sequential(*list(pretrained.children())[:-1]) DO NOT USE NN SEQUENTIAL WITH TRANSFORMERS -->
        # original functionality is lost
        # 5 head: just switch classifier to identity
        self.pretrained.heads = nn.Identity()
        # freeze the model
        for param in self.pretrained.parameters():
            param.requires_grad = False
    def forward(self, x):
        return self.pretrained(x)

class ContrastiveModel_tf(nn.Module):
    def __init__(self, embedding_dim=32, projection_dim=64, num_heads=4, num_layers=1, dropout_rate=0):
        super().__init__()
        self.eye_encoder = EyeTransformerEnc(num_layers=num_layers, num_heads=num_heads, embedding_dim=embedding_dim, dropout_rate=dropout_rate)
        self.image_encoder = ImageEncoder()
        self.projection_eye = ProjectionHead(embedding_dim=2*embedding_dim, projection_dim=projection_dim)
        self.projection_image = ProjectionHead(embedding_dim=768, projection_dim=projection_dim)
    def forward(self, batch):
        eye_input = batch["eye_data"]
        image_input = batch["img"]
        eye_features = self.eye_encoder(eye_input)
        image_features = self.image_encoder(image_input)
        eye_embedding = self.projection_eye(eye_features)
        image_embedding = self.projection_image(image_features)

        # compute similarity
        logits = eye_embedding @ image_embedding.T

        # targets
        target_eye = eye_embedding @ eye_embedding.T
        target_image = image_embedding @ image_embedding.T
        targets = torch.softmax((target_eye + target_image) / 2, dim=-1)

        # loss
        eye_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        image_loss = nn.functional.cross_entropy(logits.T, targets.T, reduction='none')

        loss = (eye_loss + image_loss) / 2

        return loss.mean()

# %% Test model
if __name__ == "__main__":
    """model = EyeTransformerEncClass(num_layers=1, num_heads=4, n_classes=1700, embedding_dim=32)

    # Test input
    
    # make float for model
    input_ids = input_ids.float()

    # Forward pass
    output = model(input_ids)
    print(output.shape)
    print(output)

    #print(model)

    # parameter count
    print("Num Params", sum(p.numel() for p in model.parameters() if p.requires_grad))"""

    input_ids = torch.randn(5, 3, 1400).float()  # Batch size 1, 3 channels, 1400 time steps

    input = torch.randn(5, 3, 224, 224).float()
    batch = (input_ids, input)
    model = ContrastiveModel_tf().to(device)
    print(model(batch))
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))