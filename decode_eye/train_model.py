import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
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

# First try
def epoch_loop(model, data, criterion, optimizer):
    is_train = False if optimizer is None else True

    # set train/eval accordingly
    model.train() if is_train else model.eval()
    with torch.set_grad_enabled(is_train):
        input_data = torch.tensor(data[0], dtype=torch.float)
        target = torch.tensor(data[1], dtype=torch.float)

        outputs = model(input_data) # forward pass

        loss = criterion(outputs, target) # loss function
        acc = accuracy(outputs, target)

        if is_train:
            optimizer.zero_grad() # reset gradients
            loss.backward() # backpropagation of loss
            optimizer.step() # update weights

    return loss, acc

# acc, lr definen


# New try
# Adjust dataloader

def train_model(model, ReadData, criterion, optimizer, device, num_epochs=10):
    model.to(device)

    # update model weights by looping through dataset multiple times
    for epoch in range(num_epochs):
        model.train() # set model to train mode
        running_loss = 0.0 # keep track of total loss
        correct, total = 0, 0 # calculate accuracy

        for inputs, labels in ReadData:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() # reset gradients
            outputs = model(inputs) # forward pass
            loss = criterion(outputs, labels)
            loss.backward() # backpropagation
            optimizer.step() # update weights

            # track loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum(),item()

        epoch_loss = running_loss / len(ReadData) # average loss over all batches in epoch
        epoch_acc = 100 * correct / total # accuracy as %
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%") # epoch-wise progress

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### has to be adjusted
# for the CNN
cnn_model = ResNet1D(in_channels=3, num_classes=10, patch_size=10)
cnn_criterion = nn.CrossEntropyLoss()
cnn_optimizer = optim.AdamW(cnn_model.parameters(), lr=0.001)

# train CNN
train_model(cnn_model, ReadData, cnn_criterion, cnn_optimizer, device, num_epochs=10)

# for transfomer model
transformer_model = EyeTransformerEncClass(embedding_dim=64, num_heads=8, num_layers=1, dropout_rate=0.1)
transformer_criterion = nn.CrossEntropyLoss()
transformer_optimizer = optim.AdamW(transformer_model.parameters(), lr=0.001)




