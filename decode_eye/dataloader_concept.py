"""
This code is a concept for the Dataloader.
"""

# load necessary libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# PyTorch Dataset Class for Simulated Data
class SimData(torch.utils.data.Dataset):
    def __init__(self, gts_path, eye_path):
        """
            Args:
            gts (str): Path to the ground truth labels .npy file.
            eye (str): Path to the trials .npy file.
        """
        self.gts = np.load(gts_path)
        self.eye = np.load(eye_path)

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
        eye_bin = self.eye[idx, :, :] # eye_bin: 3x50 array with eye data
        gt_bin = self.gts[idx]  # gt_bin = label for that bin

        # shape of gt may have to be changed to match output shape of network
        return torch.from_numpy(eye_bin), torch.from_numpy(np.array([gt_bin]))


# Create dataset
dataset = SimData(gts_path="../data/data_eye/gt_trials_preprocessed.npy", eye_path="../data/data_eye/trials_preprocessed.npy")

print(dataset.__len__())
print(dataset.__getitem__(1)[0].shape)
print(dataset.__getitem__(1)[1].shape)

# Create Data Loader with batch size of 12
data_loader_train = torch.utils.data.DataLoader(dataset=dataset, batch_size=12, shuffle=True)