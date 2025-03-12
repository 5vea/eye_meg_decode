"""This script trains a model on the eye data.

This script is adapted on the train_model.py script. Just for contrastive learning, so training loop and data loading is adapted.

We will also work with 2/3 of the labels, to have a more balanced dataset. And 1/3 left out labels for later evaluation.
Furthermore one item per training label is left out for epoch evaluation.

Dataloaders done
Model done
Training done

Check save in eval

To Do:
Evaluation
Interpretation
- kernel probing in cnn (mechanistic interpretation)
- attention maps in transformer (temporal interpretation)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-951bdf40-e7e4-368e-ca9e-d63cf9d23f30"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import numpy as np
import transformer_cl_model as tf
import cnn_cl_model as cnn
import argparse
import json
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torchvision
from PIL import Image
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def save_checkpoint(model, optimizer, scheduler, epoch, path):
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(checkpoint, os.path.join(path, 'checkpoint.pth'))
    print(f"Checkpoint saved at epoch {epoch}")



#%% setup argparse
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=50, help="Batch size.")
parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs.")
parser.add_argument("--init_lr", type=float, default=0.0003, help="Learning rate at start.")
parser.add_argument("--min_lr", type=float, default=0.00001, help="Learning rate at end of cos annealing.")
parser.add_argument("--model", type=str, default="transformer", help="Model to use.")
parser.add_argument("--bin", type=int, default=0, help="Only false implemented.")
parser.add_argument("--load", type=int, default=0, help="Load existing model.")
parser.add_argument("--model_id", type=int, default=0, help="Model id.")
parser.add_argument("--subj", type=int, default=1, help="Subject to train.")
parser.add_argument("--vit_lr", type=float, default=0.0001, help="Learning rate for the ViT model.")

args = parser.parse_args()

model_folder = "/home/saskia_fohs/mbb_dl_project/models/subj" + str(args.subj) + "/" + args.model + "_CL" + "_" + str(args.model_id) + "_" + str(args.epochs) + "_" + str(args.init_lr).replace(".", "-") + "_" + str(args.batch_size)

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
    def __init__(self, gts_path, eye_path, classes, img_transforms=None, img_path=None):
        """
        Args:
            gts_path (str): Path to the ground truth labels .npy file.
            eye_path (str): Path to the trials .npy file.
        """
        self.gts = np.load(gts_path)
        self.eye = np.load(eye_path)

        self.img_transforms = img_transforms

        self.img_path = img_path

        # correct dataset for test images
        self.corrector()

        self.classes = classes
        # for output vector
        self.label_output = np.unique(self.gts[:,0])

        if override:
            self.labels_train = np.load(model_folder + "/labels_train.npy")
        else:
            # create labels_train from two thirds of label output
            self.labels_train = np.random.choice(self.label_output, int(len(self.label_output)/3) * 2, replace=False)
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

    def get_image(self, id):
        """
        Returns the image path for a given image id.
        """
        return self.classes[self.classes["things_image_nr"] == id]["image_path"].values[0]

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
        img_id = self.gts[idx,1]

        img_spec = self.get_image(img_id)

        one_hot = self.label_output == label

        # to index
        label = np.argmax(one_hot)

        if self.img_transforms is not None:
            img = Image.open(self.img_path + "/" + img_spec)
            img = self.img_transforms(img)
        else:
            img = None

        data = {"eye_data": eye_data.astype(np.float32), "label": label, "img": img}

        return data

# %% Prep the sets for the loader
# Create dataset
data_directory = "/home/saskia_fohs/mbb_dl_project/preprocessed"
img_dir = "/home/saskia_fohs/mbb_dl_project/things_images"
eye_path = data_directory + "/trial_data_P" + str(args.subj) + ".npy"
gt_path = data_directory + "/trial_gt_P" + str(args.subj) + ".npy"
classes = pd.read_csv(data_directory + "/categories.csv")

# vit image transform
transform = torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1.transforms()

dataset = ReadData(gts_path=gt_path, eye_path=eye_path, classes=classes, img_path=img_dir, img_transforms=transform)

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
def epoch_loop(model_epoch, data, optimizer=None, scheduler=None):
    is_train = False if optimizer is None else True

    # set train/eval accordingly
    model_epoch.train() if is_train else model_epoch.eval()

    lr = None

    losses = []
    with torch.set_grad_enabled(is_train):
        for batch in data:
            batch = {k: v.to(device) for k, v in batch.items() if k != "label"}
            loss = model_epoch(batch)
            losses.extend([loss.item() for _ in range(batch["img"].size(0))])

            if is_train:
                optimizer.zero_grad() # reset gradients
                loss.backward() # backpropagation of loss
                optimizer.step() # update weights

        if is_train:
            scheduler.step() # update learning rate
            lr = scheduler.get_last_lr()[0]

    # return losses over batches
    return losses, lr

def train_model(model, dl_dict, optimizer, scheduler, num_epochs=100, epoch_start=0):
    model.to(device)

    modes = ["train", "val"]

    # update model weights by looping through dataset multiple times
    for epoch in range(epoch_start, num_epochs + epoch_start):
        for mode in modes:
            losses, lr = epoch_loop(model, dl_dict[mode], optimizer, scheduler)
            loss = np.mean(losses)
            print(f"{mode.capitalize()}: Epoch {epoch+1}/{num_epochs},\t "
                  f"Loss: {loss:.4f}")
            writer.add_scalar(f"Loss/{mode}", loss, epoch)
            if lr is not None:
                writer.add_scalar("Learning Rate", lr, epoch)

        # save checkpoint after each epoch
        save_checkpoint(model, optimizer, scheduler, epoch, model_folder)

# create model
if args.model == "cnn":
    model = cnn.ContrastiveModel_cnn()
elif args.model == "transformer":
    model = tf.ContrastiveModel_tf()
else:
    raise ValueError("Model not found.")

# parameter count
print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters.")

# different lr for different parts of the model
params = [
        #{"params": model.image_encoder.parameters(), "lr": args.vit_lr, "weight_decay": 0.},
        {"params": model.eye_encoder.parameters(), "lr": args.init_lr},
        {"params": itertools.chain(
            model.projection_eye.parameters(), model.projection_image.parameters()
        ), "lr": args.init_lr}
    ]

# optimizer
optimizer = optim.AdamW(params, weight_decay=0.01)

# cosine annealing, no restarts
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=args.min_lr)

# to device
model.to(device)

if args.load:
    start_epoch = load_checkpoint(model, optimizer, scheduler, model_folder)
else:
    start_epoch = 0

# train model
train_model(model, dataloaders, optimizer, scheduler, num_epochs=args.epochs, epoch_start = start_epoch)

writer.close()