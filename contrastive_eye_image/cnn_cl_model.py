"""
CNN CL Model
Same PreResNet just without class layer and with ViT Embedding Matching

The image encoder is a pretrained ViT model from torchvision, but its parameters are not frozen. It will have a lower learning rate than the rest of the model.
See training script for more details.
"""

# import libraries
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torchvision

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

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

class ResNet1D(nn.Module):
    """ResNet-like architecture for 1D signals including
    - initial convolutional layer
    - 3 ResNet layers with residual connections (_make_layer)
    - global average pooling for dimensionality reduction
    """
    def __init__(self, num_blocks=[2, 2, 2], channels=[64, 128, 256], dropout=0.1):
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

        # dropout
        self.dropout = nn.Dropout(dropout)

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
        x = torch.flatten(x, 1) # (batch_size, channels)

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
        """# freeze the model
        for param in self.pretrained.parameters():
            param.requires_grad = False"""
    def forward(self, x):
        return self.pretrained(x)

class ContrastiveModel_cnn(nn.Module):
    def __init__(self, projection_dim = 64,num_blocks=[1, 1, 1], channels=[64, 128, 256], dropout=0):
        super().__init__()
        self.eye_encoder = ResNet1D(num_blocks=num_blocks, channels=channels, dropout=dropout)
        self.image_encoder = ImageEncoder()
        self.projection_eye = ProjectionHead(embedding_dim=channels[-1], projection_dim=projection_dim)
        self.projection_image = ProjectionHead(embedding_dim=768, projection_dim=projection_dim)
    def forward(self, batch):
        eye_input = batch[0]
        image_input = batch[1]
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

if __name__ == "__main__":
    # create model
    model = ContrastiveModel_cnn()

    # test model with random input
    x = torch.randn(5, 3, 1401)
    y = torch.randn(5, 3, 224, 224)

    print(model([x, y]))

    # prnt parameters
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))