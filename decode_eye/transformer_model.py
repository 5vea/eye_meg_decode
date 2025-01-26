"""
This code is for the transformer model
In principle it is similar to the ViT logic: first we want to build patches from our eye data
Then we want to tokenize them with kernels --> the dimension of the patch embedding is therefore equal to the number of kernels
--> one kernel encodes the patch in one dimension
Opposed to the ViT we will not use a CLS token, because this model is for pure classification, which means that we will not do
any pretraining for which a "patch wise" output is needed
To keep things more easily interpretable, we will aim for a simple model, with only one transformer layer and only some MLP layers
Later we want to probe the attention maps of each head
Self attention will be used, but there will be no masking: we do not want to inhibit the direction of interaction between the patches

CLS vs FC Classifier?
- this question kept us a bit up at night
- CLS tokens are especially useful, because here we have a sequence to sequence encoder -> so adding a CLS token as just a
part of the sequence keeps things flexible, because the length of the sequence does not need to be fixed
- when using an FC classifier on all context embeddings, we need a fixed patch length, because the dimension
of the layer needs to be fixed

--> for the sake of simplicity, we will for now settle on an FC classifier on the flattened context embeddings
for cls discussion, maybe consider global average pooling with adapted LR
https://datascience.stackexchange.com/questions/90649/class-token-in-vit-and-bert

I will base the class on https://deeplearning-jupyterbook.github.io/notebooks/llm.html

The positional encoding with i, sin and cos will be used
"""

# maybe add skip connections in attention head

import numpy as np
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchtune.modules import RotaryPositionalEmbeddings
import math

# put in argument parser
# currently for 100 ms bins
n_classes = 10
patch_count = 10
patch_size = 10

# utils to save checkpoints
# look at deepcsf for it --> always save state dict of optimizer, network and scheduler, if you did it

# tensorboard logging + close tb am ende
# look at deepcsf for logging and arg parsing

# positional encoding from https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch
# set dropout to zero, because it is no learnable embedding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0, max_len = 10):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

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
        return self.dropout(x)

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

        """# Apply causal mask to prevent attention to future tokens
        attention_scores = attention_scores.masked_fill(self.tril[:sequence_length, :sequence_length] == 0,
                                                        float('-inf'))"""

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
class EyeTransformerEncClass(nn.Module):
    """A GPT-based language model that utilises transformer blocks to generate sequences."""

    def __init__(self, embedding_dim=64, num_heads=12, num_layers=1, dropout_rate=0):
        """
        Initialises the model with specified vocabulary size, embedding dimension, number of heads,
        number of transformer layers, and dropout rate.

        Args:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of token embeddings.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        dropout_rate (float): Dropout probability for regularisation.
        """
        super().__init__()

        # exclusive convolution to project to embedding space
        # here we are just moving in time: 3 channels = x,y and pupil dilation
        self.conv_projection = nn.Conv1d(in_channels=3, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size)

        # RoPE embedding
        #self.position_embedding = RotaryPositionalEmbeddings(dim=embedding_dim, max_seq_len=patch_count)
        # --> did not work for us
        self.position_embedding = PositionalEncoding(embedding_dim)

        # Stack of transformer blocks
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(embedding_dim, num_heads=num_heads, dropout_rate=dropout_rate) for _ in
              range(num_layers)]
        )

        # Final layer normalisation for stable outputs
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

        # add one MLP non linearity
        self.mlp_head = MLP(embedding_dim, dropout_rate)

        # Output layer FC on all context embeddings
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * patch_count, 2 * embedding_dim * patch_count),  # expects flattened FC input
            nn.GELU(),  # swapped to GeLU
            nn.Linear(2 * embedding_dim * patch_count, n_classes)  # for classes
        )

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
        Tensor: Logits of shape (batch_size, sequence_length, vocab_size) indicating probabilities of each token.
        """
        batch_size, channels, sequence_length = input_ids.shape

        # Create token embeddings
        token_embeddings = self.conv_projection(input_ids)  # Shape: (batch_size, embedding_dim, sequence_length)
        # warning: dim needs to be swapped

        # this is annoying, but embedding dim and channel out need to be switched
        token_embeddings = torch.swapaxes(token_embeddings, 1,2)
        # now it is on normal grounds, lets make it ugly for the function
        token_embeddings = torch.swapaxes(token_embeddings, 0, 1)

        x = self.position_embedding(token_embeddings)

        # switch back to normal axes
        x = torch.swapaxes(x, 0, 1)

        # Pass through stacked transformer blocks
        x = self.transformer_blocks(x)  # Shape: (batch_size, sequence_length, embedding_dim)

        # Apply final layer normalisation
        x = self.final_layer_norm(x)  # Shape: (batch_size, sequence_length, embedding_dim)

        # MLP head
        #x = self.mlp_head(x)

        # flatten after batch size for FC classifier
        x = torch.flatten(x, start_dim=1)

        # Convert to logits for each token in the vocabulary
        logits = self.classifier(x)  # Shape: (batch_size, sequence_length * patch_count)

        return logits
    # cls token for classification

# missing: skip connections!

# for CNN just orient on resnet strucutre to have it adaptable
## --> alle klassen sind fein --> bro alter, das ist ja mal ein ding