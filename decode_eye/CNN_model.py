"""
This code is for the CNN model.
We tried to create a simple ResNet model.
"""

# import libraries
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torchvision


#DAS IST ALLES NUR GEKLAUT EYOEYO ICH HAB NEN HOLZKOPF UND AUCH STEINE, EYO!
#DAS IST ALLES NUR GEKLAUT EYOEYO, DOCH DAS WEIß ICH NUR UND DIE SALZIKAI GANZ ALLEINE, EYO!
#DAS IST ALLES NUR GEKLAUT, NUR GEZOGEN, NUR GESTOHLEN UND GERAUBT
#TSCHULDIGUNG, HAB MIR MEIN EIGENES GRAB GEBAUT
#TSCHULDIGUNG, MIT VERLAUB
#<'3

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Architecture of ResNet-like model
"""Basic Building Block that applies two convolutional layers with batch normalization and ReLU activation in between. 
Layer outputs are added to input of the block (residual connection)
If input and output shapes differ, an adjustment layer adjusts the dimensions of the input for the addition"""
class BasicBlock(nn.Module):
    """Define the basic res block"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False) # first conv layer
        self.bn1 = nn.BatchNorm1d(out_channels) # batch normalization after conv1
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False) # second conv layer
        self.bn2 = nn.BatchNorm1d(out_channels) # batch normalization after conv2
        self.downsample = downsample # adjustment layer when input and output shapes mismatch

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out)

        return out

class ResNet1D(nn.Module):
    """ResNet-like architecture for 1D signals including
    - initial convolutional layer
    - 3 ResNet layers with residual connections (_make_layer)
    - global average pooling for dimensionality reduction
    - fully connected output layer for classification"""
    def __init__(self, in_channels, num_classes, patch_size, num_blocks=[2, 2, 2], channels=[64, 128, 256]):
        super(ResNet1D, self).__init__()

        self.in_channels = channels[0]
        self.conv1 = nn.Conv1d(in_channels, channels[0], kernel_size=7, stride=patch_size, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(channels[0])
        self.relu = nn.ReLU(inplace=True)

        # layers for ResNet
        self.layer1 = self._make_layer(channels[0], channels[0], num_blocks[0])
        self.layer2 = self._make_layer(channels[0], channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(channels[1], channels[2], num_blocks[2], stride=2)

        # global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[2], num_classes) # classifier

    """Creating sequence of multiple BasicBlocks to form ResNet layer
    First layer might use adjustment if input size changes (also z.B. stride)
    The subsequent blocks maintain the same nr of output channels"""

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        """Create ResNet layer with specified nr of blocks"""
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

            layers = []
            layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
            self.in_channels = out_channels

            for _ in range(1, num_blocks):
                layers.append(BasicBlock(self.in_channels, out_channels))

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.global_pool(x) # (batch_size, channels, 1)
            x = torch.flatten(x, 1)
            x = self.fc(x)

            return x

# Hyperparameters
in_channels = 3 # nr of input channels (x, y, pupil dil)
num_classes = 10 # nr of output classes
patch_size = 10

model = ResNet1D(in_channels=in_channels, num_classes=num_classes, patch_size=patch_size)
print(model)