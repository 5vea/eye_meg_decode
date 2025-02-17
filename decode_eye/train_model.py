import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import transformer_model as tf
import CNN_model as cnn
import argparse
import os
import json

#os.environ["CUDA_VISIBLE_DEVICES"] = ""

#%% setup argparse
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=10, help="Batch size.")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
parser.add_argument("--lr", type=float, default=0.002, help="Learning rate.")
parser.add_argument("--model", type=str, default="cnn", help="Model to use.")
parser.add_argument("--bin", type=int, default=0, help="Only false implemented.")
parser.add_argument("--load", type=int, default=0, help="Load existing model.")
parser.add_argument("--model_id", type=int, default=0, help="Model id.")

args = parser.parse_args()

model_folder = "/home/saskia_fohs/mbb_dl_project/models/" + args.model + "_" + str(args.model_id) + "_" + str(args.epochs) + "_" + str(args.lr).replace(".", "-") + "_" + str(args.batch_size)

print(args.model)
print(args.load)

override = False

# check if load is true
if args.load:
    print("Load existing model.")
    try:
        # load model
        #model = torch.load(model_folder + "/model.pt")
        with open(model_folder + "/model_params.json", "r") as f:
            args.__dict__ = json.load(f)
        override = True
    except FileNotFoundError:
        print("Model not found. Train new model.")
        args.load = False
print(args.load)
# needs extra if, because if load fails, we need to train a new model
if (not args.load) and (not override):
    print("Train new model.")
    # if already exists, skip
    try:
        os.mkdir(model_folder)
    except FileExistsError:
        raise FileExistsError("Model already exists. Change model_id.")
    # create model
    """if args.model == "cnn":
        model = cnn.ResNet1D()
    elif args.model == "transformer":
        model = tf.EyeTransformerEncClass()
    else:
        print("Model not found. Train new model.")
        raise ValueError("Model not found.")"""
    #model_params = model.get_params()
    # save model params
    with open(model_folder + "/model_params.json", "w") as f:
        json.dump(args.__dict__, f, indent=2)

# %% Dataset
class ReadData(torch.utils.data.Dataset):
    def __init__(self, gts_path, eye_path, classes):
        """
            Args:
            gts_path (str): Path to the ground truth labels .npy file.
            eye_path (str): Path to the trials .npy file.
        """
        self.gts = np.load(gts_path)
        self.eye = np.load(eye_path)
        self.classes = classes
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
        eye_data = self.eye[idx]
        label = self.gts[idx]

        return eye_data, label