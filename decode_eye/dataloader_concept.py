"""
This code is a concept for the Dataloader.
"""

# load necessary libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# %% Function to plot the eye data
# Maybe later we can move this as utility in a separate file or as a method in the dataset class
def plot_bin(data):
    """
    Args:
        data (np.array): 3x50 array with eye data.
    """
    # create a time vector (bin size = 50 ms, sampled at 1 kHz)
    time = np.linspace(0, 50, data.shape[1])  # time in ms

    # Plot x, y, and p underneath each other
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)  # 3 rows, 1 column, shared x-axis

    # list for ax properties
    ax_ylabel = ["x (movement)", "y (movement)", "p (dilation)"]
    ax_labels = ["x", "y", "p"]
    ax_colors = ["blue", "green", "red"]

    # Plot each signal (in rows)
    for ax_id, ax in enumerate(axes):
        ax.plot(time, data[ax_id, :], label=ax_labels[ax_id], color=ax_colors[ax_id])
        ax.set_ylabel(ax_ylabel[ax_id])
        ax.legend(loc="upper right")

    # Adjust layout
    plt.tight_layout()
    plt.show()

# %% PyTorch Dataset Class for Simulated Data
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
        gt_bin = self.gts[idx]  # gt_bin: label for that bin

        # where gt_bin is equal to the label_output, return 1, else 0; this is the output vector
        gt_y = self.label_output == gt_bin # gt_y: 4x1 array with 1 at the index of the label, 0 elsewhere

        # return x and y as torch tensors
        return torch.from_numpy(eye_bin), torch.from_numpy(gt_y.astype(np.int32))

# %% Test the Dataloader
# Create dataset
# passing paths might be changed for later split before dataset init
dataset = ReadData(gts_path="../data/data_eye/gt_trials_preprocessed.npy", eye_path="../data/data_eye/trials_preprocessed.npy")

print(dataset.__len__())
print(dataset.__getitem__(1)[0].shape)
print(dataset.__getitem__(1)[1].shape)
print(dataset.__getitem__(1)[1])

# Create Data Loader with batch size of 12
data_loader_train = torch.utils.data.DataLoader(dataset=dataset, batch_size=12, shuffle=True)

# get one sample from our dataset
eye_bin, gt_y = dataset.__getitem__(1)

# convert eye_bin tensor to numpy array for plotting and call plotting function
plot_bin(eye_bin.numpy())