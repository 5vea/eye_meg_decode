"""
This code is for the transformer model
We will combine multiple ideas: a cls token and GAP will be used as input for the final classification layers.
--> the model can therefore just use both and decide which one to use by assigning weights
In principle we have a transformer encoder with a final classification layer
--> GAP will be used on the context embeddings of the sequence and cls token will be concatenated to the averaged context embeddings

The input will be projected by a convolutional layer of 1x1 convolutions to the embedding space. Our original signal has 3 channels (x, y, pupil dilation).
The positional encoding will be added to the input embeddings.

If the transformer is training, it will apply smooth time masking to the input data, so that the model is hindered to
learn the exact timepoints of the data. This is done by smoothly masking the inputs in the time dimension.
Also, amplitude jitter is applied to the input data, so that the model is not able to learn the exact amplitudes of the data.
The amplitude jitter is additive, because signal inversions would distort later interpretation techniques.
"""

import numpy as np
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# utils to save checkpoints
# look at deepcsf for it --> always save state dict of optimizer, network and scheduler, if you did it

# tensorboard logging + close tb am ende
# look at deepcsf for logging and arg parsing

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

class SmoothTimeMask(nn.Module):
    """Applies a smooth time mask to the input data during training. From https://arxiv.org/pdf/2206.14483
    Warning: had to correct their formula: the input for the second sigmoid needs to be t - mask_start - mask_length and
    not mask_start + mask_length - t
    Furthermore, the first sigmoid needs to be mask_start - t and not t - mask_start
    Maybe text authors about it."""

    def __init__(self, mask_prob=0.1, mask_span=0.1, transition=0.3):
        """
        Args:
        mask_prob (float): Probability of masking a time span.
        mask_span (float): Maximum proportion of the sequence to mask.
        transition (float): Transition width for the sigmoid function.
        """
        super().__init__()
        self.mask_prob = mask_prob
        self.mask_span = mask_span
        self.transition = transition

    def sigmoid(self, t):
        return 1 / (1 + torch.exp((-t) * self.transition))

    def mask_sigmoid(self, t, mask_start, mask_length):
        return self.sigmoid(mask_start - t) + self.sigmoid(t - mask_start - mask_length)

    def forward(self, x):
        batch_size, channels, sequence_length = x.shape

        # decide if item in batch is masked
        mask = torch.rand(batch_size) < self.mask_prob

        # get points to mask
        mask_start = torch.randint(0, int(sequence_length * (1 - self.mask_span)), (batch_size,)).unsqueeze(1)
        mask_length = sequence_length * self.mask_span

        x[mask,:,:] = x[mask,:,:] * self.mask_sigmoid(torch.arange(sequence_length), mask_start, mask_length).unsqueeze(1)[mask,:,:].to(device)

        return x

class AmplitudeJitter(nn.Module):
    """Applies gaussian noise to the input data during training."""

    def __init__(self, jitter_factor=0.1):
        """
        Args:
        jitter_factor (float): Maximum amplitude scaling factor.
        """
        super().__init__()
        self.jitter_factor = jitter_factor

    def forward(self, x):
        batch_size, sequence_length, embedding_dim = x.shape

        # generate gaussian noise
        noise = torch.randn(batch_size, sequence_length, embedding_dim) * self.jitter_factor

        # add noise to input
        return x + noise.to(device)


# needs to be adapted
class EyeTransformerEncClass(nn.Module):
    """ A transformer encoder model for eye data classification. """

    def __init__(self, embedding_dim=64, num_heads=12, num_layers=1, dropout_rate=0, n_classes=1000):
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

        # add one MLP non linearity
        self.mlp_head = MLP(embedding_dim, dropout_rate)

        # Output layer FC on all context embeddings
        # Size is reasonable because there are a lot of classes, so we need a lot of parameters
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, 2 * 2 * embedding_dim),  # expects flattened FC input
            nn.GELU(),  # swapped to GeLU
            nn.Linear(2 * 2 * embedding_dim, n_classes)  # for classes
        )

        # Initialise weights for stability
        self.apply(self._init_weights)

        # Reg Data Aug
        self.time_mask = SmoothTimeMask(mask_prob=0.3, mask_span=0.07, transition=9)
        self.amplitude_jitter = AmplitudeJitter(jitter_factor=0.05)

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

        # apply data augmentation only during training
        if self.training:
            input_ids = self.time_mask(input_ids)
            input_ids = self.amplitude_jitter(input_ids)

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

        # flatten after batch size for FC classifier
        x = torch.flatten(x, start_dim=1)

        # make prediction
        logits = self.classifier(x)  # Shape: (batch_size, n_classes)

        return logits


# %% Test model
if __name__ == "__main__":
    model = EyeTransformerEncClass(num_layers=1, num_heads=4, n_classes=1700, embedding_dim=32)

    # Test input
    input_ids = torch.randint(1, 100, (2, 3, 20))  # Batch size 1, 3 channels, 1400 time steps
    # make float for model
    input_ids = input_ids.float()

    # Forward pass
    output = model(input_ids)
    print(output.shape)
    print(output)

    #print(model)

    # parameter count
    print("Num Params", sum(p.numel() for p in model.parameters() if p.requires_grad))