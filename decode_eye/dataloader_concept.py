"""
This code is a concept for the Dataloader.
"""

# load necessary libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# PyTorch Dataset Class for Simulated Data
class ReadData(torch.utils.data.Dataset):
    def __init__(self, gts_path, eye_path):
        """
            Args:
            gts_path (str): Path to the ground truth labels .npy file.
            eye_path (str): Path to the trials .npy file.
        """
        self.gts = np.load(gts_path)
        self.eye = np.load(eye_path)
        # for output vector
        self.label_output = np.unique(self.gts)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.eye)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample.
        """
        # Get data
        eye_bin = self.eye[idx, :, :] # eye_bin: 3x50 array with eye data
        gt_bin = self.gts[idx]  # gt_bin = label for that bin

        # where gt_bin is equal to the label_output, return 1, else 0; this is the output vector
        gt_y = self.label_output == gt_bin # gt_y: 4x1 array with 1 at the index of the label, 0 elsewhere

        # return x and y as torch tensors
        return torch.from_numpy(eye_bin), torch.from_numpy(gt_y.astype(np.int32))


# Create dataset
# passing paths might be changed for later split before dataset init
dataset = ReadData(gts_path="../data/data_eye/gt_trials_preprocessed.npy", eye_path="../data/data_eye/trials_preprocessed.npy")

print(dataset.__len__())
print(dataset.__getitem__(1)[0].shape)
print(dataset.__getitem__(1)[1].shape)
print(dataset.__getitem__(1)[1])

# Create Data Loader with batch size of 12
data_loader_train = torch.utils.data.DataLoader(dataset=dataset, batch_size=12, shuffle=True)