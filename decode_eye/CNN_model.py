"""
This code is for the CNN model.
We tried to create a simple PreResNet model.

The model is a PreResNet-like architecture for 1D signals.
To improve performance and to be able to apply dropout that was tested at different positions in the same way as https://arxiv.org/pdf/2302.06112
we made the residual blocks after PreResNet (https://arxiv.org/pdf/1603.05027), instead of the original ResNet architecture.

The head is similar to MobileNet.

To avoid overfitting, we applied data augmentation techniques during training. We used a SmoothTimeMask and AmplitudeJitter to add noise to the input data.
Those are applied only during training and described more in detail in the transformer_model.py file.

Furthermore, we added dropout to the PreResNet blocks and the fully connected layer, as proposed by https://arxiv.org/pdf/2302.06112
Dropout was applied after the last bn but before the second weight layer in a BasicBlock.
Furthermore, dropout was applied in the head after the last batch norm but before the GAP.
"""

# import libraries
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Architecture of ResNet-like model
"""Basic Building Block that applies two convolutional layers with batch normalization and ReLU activation in between. 
Layer outputs are added to input of the block (residual connection)
If input and output shapes differ, an adjustment layer adjusts the dimensions of the input for the addition

PreResNet Structure.
"""
class BasicBlock(nn.Module):
    """Define the basic res block"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout=0.1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channels) # batch normalization before conv1
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False) # first conv layer
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False) # second conv layer
        self.bn2 = nn.BatchNorm1d(out_channels) # batch normalization after conv2
        self.relu = nn.ReLU(inplace=False) # ReLU activation function
        self.downsample = downsample # adjustment layer when input and output shapes mismatch

        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)

        # dropout
        out = self.dropout(out)

        out = self.relu(out)
        out = self.conv2(out)

        out += identity

        return out

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

class ResNet1D(nn.Module):
    """ResNet-like architecture for 1D signals including
    - initial convolutional layer
    - 3 ResNet layers with residual connections (_make_layer)
    - global average pooling for dimensionality reduction
    - fully connected output layer for classification"""
    def __init__(self, num_classes, num_blocks=[2, 2, 2], channels=[64, 128, 256], dropout=0.1):
        super(ResNet1D, self).__init__()

        self.droprate = dropout

        self.conv1 = nn.Conv1d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(channels[0])
        self.relu = nn.ReLU(inplace=True)

        # layers for ResNet
        self.layer1 = self._make_layer(channels[0], channels[0], num_blocks[0])
        self.layer2 = self._make_layer(channels[0], channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(channels[1], channels[2], num_blocks[2], stride=2)

        self.bn2 = nn.BatchNorm1d(channels[2]) # batch norm before classifier

        # global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[2], num_classes) # classifier

        # dropout
        self.dropout = nn.Dropout(dropout)

        # Reg Data Aug
        self.time_mask = SmoothTimeMask(mask_prob=0.3, mask_span=0.07, transition=9)
        self.amplitude_jitter = AmplitudeJitter(jitter_factor=0.05)

    """Creating sequence of multiple BasicBlocks to form ResNet layer
    First layer might use adjustment if input size changes (also z.B. stride)
    The subsequent blocks maintain the same nr of output channels"""

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        """Create ResNet layer with specified nr of blocks"""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample, dropout=self.droprate))
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(BasicBlock(in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # apply data augmentation only during training
        if self.training:
            x = self.time_mask(x)
            x = self.amplitude_jitter(x)

        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn2(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.global_pool(x) # (batch_size, channels, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    # create model
    model = ResNet1D(num_classes=4, num_blocks=[1, 1, 1], channels=[64, 128, 256])
    print(model)
    # prnt parameters
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))