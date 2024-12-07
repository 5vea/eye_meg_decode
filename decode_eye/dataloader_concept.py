"""
This code is a concept for the Dataloader.
"""

# load necessary libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# load preprocessed data
gts_data = np.load("../data/data_eye/gt_trials_preprocessed.npy", allow_pickle=True)
eye_data = np.load("../data/data_eye/trials_preprocessed.npy")

print(gts_data.shape)
print(eye_data.shape)

# PyTorch Dataset Class for Simulated Data
class SimData(torch.utils.data.Dataset):
    def __init__(self, gts, eye):
        """
        Initialise the dataset with images, ground-truth labels, and optional transformations.

        Args:
            gts (str): Path to the ground truth labels .npy file.
            eye (str): Path to the trials .npy file.
        """
        self.gts = gts
        self.eye = eye

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.eye)

    def __getitem__(self, idx):
        """
        Retrieves a sample and its ground-truth label by index.

        Args:
            idx (int): Index of the sample.
        """
        # Get data
        eye = self.eye[idx, :, :]
        gts = self.gts[idx]  # gt = ground truth

        return torch.from_numpy(eye), torch.from_numpy(gts)


# Create dataset
dataset = SimData(gts=gts_data, eye=eye_data)

print(dataset.__len__())

# Create Data Loader with batch size of 12
data_loader_train = torch.utils.data.DataLoader(dataset=dataset, batch_size=12, shuffle=True)