"""
This script trains a model on the eye data.

It is not the aim to find the best possible classifier, which is why we will use a simple CNN and a simple transformer model.
Furthermore, due to data shortage, we will only use a subset of the labels for training (1/3 random per subject).
The model will have a test set comprising of one image per label, which will be used to evaluate the model and check for overfitting.

Also for the loss function, the classes will be weighted, so that the model is not biased towards the majority class.
We will use a cross entropy loss function with logarithmic weights, to give more weight to the minority classes, but not too much.

Cosine annealing will be used to update the learning rate, which will start at init_lr and end at min_lr. The scheduler will be updated after each epoch.
Using it to avoid overfitting and to find a good learning rate. We will not use restarts, because those might lead to more overfitting.
"""

"""
To Do:
better folder management!
tensorboard - load split
epoch start from last epoch
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-f4bd4d61-6e02-6347-559f-dc9c6528303e"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import transformer_model as tf
import CNN_model as cnn
import argparse
import json
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% setup argparse
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=15, help="Batch size.")
parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs.")
parser.add_argument("--init_lr", type=float, default=0.0003, help="Learning rate at start.")
parser.add_argument("--min_lr", type=float, default=0.00001, help="Learning rate at end of cos annealing.")
parser.add_argument("--model", type=str, default="transformer", help="Model to use.")
parser.add_argument("--bin", type=int, default=0, help="Only false implemented.")
parser.add_argument("--load", type=int, default=0, help="Load existing model.")
parser.add_argument("--model_id", type=int, default=0, help="Model id.")
parser.add_argument("--subj", type=int, default=1, help="Subject to train.")

args = parser.parse_args()

model_folder = "/home/saskia_fohs/mbb_dl_project/models/subj" + str(args.subj) + "/" + args.model + "_" + str(args.model_id) + "_" + str(args.epochs) + "_" + str(args.init_lr).replace(".", "-") + "_" + str(args.batch_size)

print(args.model)

override = False

# args.load to bool
args.load = bool(args.load)

print(args.load)

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
    #model_params = model.get_params()
    # save model params
    with open(model_folder + "/model_params.json", "w") as f:
        json.dump(args.__dict__, f, indent=2)

# needs load split
writer = SummaryWriter(model_folder)

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

        # correct dataset for test images
        self.corrector()

        self.classes = classes
        # for output vector
        self.label_output = np.unique(self.gts[:,0])

        if override:
            self.labels_train = np.load(model_folder + "/labels_train.npy")
        else:
            # create labels_train from one third of label output
            self.labels_train = np.random.choice(self.label_output, int(len(self.label_output)/3), replace=False)
            np.save(model_folder + "/labels_train.npy", self.labels_train)

        # only use chosen labels for training
        self.eye = self.eye[np.isin(self.gts[:,0], self.labels_train)]
        self.gts = self.gts[np.isin(self.gts[:,0], self.labels_train)]
        self.label_output = np.unique(self.gts[:,0])

    def corrector(self):
        """
        Corrects the dataset for the amount of images. Deletes images, that were displayed more than 2 times.
        :return:
        """
        # delete test images
        count_df = pd.DataFrame(self.gts[:,1]).value_counts()
        mask_df = count_df < 3
        count_df = count_df[mask_df]
        usable_images = count_df.to_frame().index.to_frame().to_numpy()[:,0]

        # final mask
        mask = np.isin(self.gts[:,1], usable_images)
        self.eye = self.eye[mask]
        self.gts = self.gts[mask]

    def train_test_split(self):
        """
        Returns the indeces of the training and test set. The test set is one image per label, chosen randomly, if load is false.
        If load is true, the indeces are loaded from the model folder.
        """
        pass
        if args.load:
            train_idx = np.load(model_folder + "/train_idx.npy")
            test_idx = np.load(model_folder + "/test_idx.npy")
        else:
            train_idx = []
            test_idx = []
            for label in self.label_output:
                label_idx = np.where(self.gts[:,0] == label)[0]
                test_idx.append(np.random.choice(label_idx, 1))
                # delete test index (number) from label_idx
                label_idx = np.delete(label_idx, np.where(label_idx == test_idx[-1]))
                train_idx.extend(label_idx)
            train_idx = np.array(train_idx).squeeze()
            test_idx = np.array(test_idx).squeeze()
            np.save(model_folder + "/train_idx.npy", train_idx)
            np.save(model_folder + "/test_idx.npy", test_idx)

        return train_idx, test_idx

    def class_weights(self):
        """
        Returns the log class weights for the loss function.
        """
        # get the amount of samples per class
        class_weights = np.zeros(len(self.label_output))
        for i, label in enumerate(self.label_output):
            class_weights[i] = len(self.gts[self.gts[:,0] == label])
        # calculate the weights logged
        class_weights = 1.0 / np.log(1.1 + class_weights)
        class_weights = torch.tensor(class_weights / class_weights.sum(), dtype=torch.float32)
        return class_weights

    def get_class(self, id):
        return self.classes[self.classes["things_category_nr"] == id]["category"].values[0]

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
        label = self.gts[idx,0]

        one_hot = self.label_output == label

        # to index
        label = np.argmax(one_hot)

        return eye_data, label

# %% Prep the sets for the loader
# Create dataset
data_directory = "/home/saskia_fohs/mbb_dl_project/preprocessed"
eye_path = data_directory + "/trial_data_P" + str(args.subj) + ".npy"
gt_path = data_directory + "/trial_gt_P" + str(args.subj) + ".npy"
classes = pd.read_csv(data_directory + "/categories.csv")

dataset = ReadData(gts_path=gt_path, eye_path=eye_path, classes=classes)

# get class weights to balance the loss function -- not super clean, because call before train_test_split, but subset
# does not inherit the class weights
class_weights_training = dataset.class_weights()

# split the dataset into training and test set
train_idx, test_idx = dataset.train_test_split()
dataset_test = torch.utils.data.Subset(dataset, test_idx)
dataset_train = torch.utils.data.Subset(dataset, train_idx)

train_dl = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True) # drop last to avoid batch size 1
test_dl = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False)

print("Training set size: ", len(dataset_train))
print("Test set size (and also n_labels): ", len(dataset_test))

# make dl dict
dataloaders = {"train": train_dl, "val": test_dl}


# %% Training -- might be modularized later
def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    Args:
        output: model output
        target: target labels
        topk: tuple of top k accuracies to calculate
    Returns:
        res: list of top k accuracies
    """

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# First try
def epoch_loop(model_epoch, data, criterion, optimizer=None, scheduler=None):
    is_train = False if optimizer is None else True

    # set train/eval accordingly
    model_epoch.train() if is_train else model_epoch.eval()
    with torch.set_grad_enabled(is_train):
        running_loss = torch.zeros(1).to(device)
        running_accs = torch.zeros(3).to(device)
        count_batches = 0
        for input, target in data:
            # convert to torch float
            input = input.float()

            input, target = input.to(device), target.to(device)

            outputs = model_epoch(input) # forward pass

            loss = criterion(outputs, target) # loss function
            acc = accuracy(outputs, target, topk=(1,5,100))

            if is_train:
                optimizer.zero_grad() # reset gradients
                loss.backward() # backpropagation of loss
                optimizer.step() # update weights
            else:
                running_loss += loss
                running_accs += torch.tensor(acc).to(device)

            count_batches += 1

        if is_train:
            scheduler.step() # update learning rate

    # return loss and accuracy, depending on train or eval
    if is_train:
        return loss, acc[0], acc[1], acc[2]
    else:
        running_loss /= torch.tensor(count_batches).to(device)
        running_accs /= torch.tensor(count_batches).to(device)
        return running_loss, running_accs[0], running_accs[1], running_accs[2]

def train_model(model, dl_dict, criterion, optimizer, scheduler, num_epochs=100):
    model.to(device)

    # update model weights by looping through dataset multiple times
    for epoch in range(num_epochs):
        epoch_loss, epoch_acc, epoch_acc_5, epoch_acc_100 = epoch_loop(model, dl_dict["train"], criterion, optimizer, scheduler)
        print(f"Train: Epoch {epoch+1}/{num_epochs},\t "
              f"Loss: {float(epoch_loss.cpu().detach().numpy()):.4f},\t "
              f"Acc: {float(epoch_acc.cpu().detach().numpy()):.2f}%,\t "
              f"Acc@5: {float(epoch_acc_5.cpu().detach().numpy()):.2f}%,\t "
              f"Acc@100: {float(epoch_acc_100.cpu().detach().numpy()):.2f}%")
        writer.add_scalar("Loss/train", float(epoch_loss.cpu().detach().numpy()), epoch)
        writer.add_scalar("Acc/train", float(epoch_acc.cpu().detach().numpy()), epoch)
        writer.add_scalar("Acc@5/train", float(epoch_acc_5.cpu().detach().numpy()), epoch)
        writer.add_scalar("Acc@100/train", float(epoch_acc_100.cpu().detach().numpy()), epoch)
        epoch_loss, epoch_acc, epoch_acc_5, epoch_acc_100 = epoch_loop(model, dl_dict["val"], criterion, None, scheduler)
        print(f"Test: Epoch {epoch + 1}/{num_epochs},\t "
              f"Loss: {float(epoch_loss.cpu().detach().numpy()):.4f},\t "
              f"Acc: {float(epoch_acc.cpu().detach().numpy()):.2f}%,\t "
              f"Acc@5: {float(epoch_acc_5.cpu().detach().numpy()):.2f}%,\t "
              f"Acc@100: {float(epoch_acc_100.cpu().detach().numpy()):.2f}% (all running)")
        writer.add_scalar("Loss/test", float(epoch_loss.cpu().detach().numpy()), epoch)
        writer.add_scalar("Acc/test", float(epoch_acc.cpu().detach().numpy()), epoch)
        writer.add_scalar("Acc@5/test", float(epoch_acc_5.cpu().detach().numpy()), epoch)
        writer.add_scalar("Acc@100/test", float(epoch_acc_100.cpu().detach().numpy()), epoch)

        # save checkpoint after each epoch
        save_checkpoint(model, optimizer, scheduler, epoch, path)

# create model
if args.model == "cnn":
    model = cnn.ResNet1D(in_channels=3, num_classes=len(dataset.label_output), patch_size=3)
elif args.model == "transformer":
    model = tf.EyeTransformerEncClass(num_layers=1, num_heads=4, n_classes=len(dataset.label_output), embedding_dim=32, dropout_rate=0.3)
else:
    raise ValueError("Model not found.")

# parameter count
print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters.")

criterion = nn.CrossEntropyLoss(weight=class_weights_training).to(device)
optimizer = optim.AdamW(model.parameters(), lr=args.init_lr, weight_decay=0.01)

# cosin annealing, no restarts
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=args.min_lr)

# to device
model.to(device)

# train model
train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=args.epochs)

writer.close()


# save model, optimizer, scheduler
path = model_folder

def save_checkpoint(model, optimizer, scheduler, epoch, path):
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(checkpoint, os.path.join(path, 'checkpoint.pth'))
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(model, optimizer, scheduler, path):
    checkpoint_path = os.path.join(path, 'checkpoint.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded from epoch {start_epoch}")
        return start_epoch
    return 0

