"""
Task Creation

In this script we create the test set for the zero-shot evaluation with the left out trials.
We will create a file for the testing trials.

Also there are 2 conditions:
1. The false label is different from the true label.
2. The false label is the same as the true label.
--> with this we can check if the model can distinguish between the two conditions.

Also we only have 5000 test trials, because we do not want to test all possible combinations, but also dont want to draw
the same trial twice.

It is optimized for pairwise comparison, to have an easier evaluation.
"""

import numpy as np
import pandas as pd
import os
import torch
import sys
sys.path.insert(0, '../..')
import transformer_cl_model as tf
import cnn_cl_model as cnn
import pickle
from tqdm import tqdm


def corrector(gt, eye):
    """
    Corrects the dataset for the amount of images. Deletes images, that were displayed more than 2 times.
    :return:
    """
    # delete test images
    count_df = pd.DataFrame(gt[:, 1]).value_counts()
    mask_df = count_df < 3
    count_df = count_df[mask_df]
    usable_images = count_df.to_frame().index.to_frame().to_numpy()[:, 0]

    # final mask
    mask = np.isin(gt[:, 1], usable_images)
    return gt[mask], eye[mask]

# model folder
path = "/home/saskia_fohs/mbb_dl_project/models/subj1/transformer_CL_2_1000_0-0003_50"
subj = 1

# load trained set
train_idx = np.load(path + "/train_idx.npy")
labels_train = np.load(path + "/labels_train.npy")

# Create dataset
data_directory = "/home/saskia_fohs/mbb_dl_project/preprocessed"
img_dir = "/home/saskia_fohs/mbb_dl_project/things_images"
eye_path = data_directory + "/trial_data_P" + str(subj) + ".npy"
gt_path = data_directory + "/trial_gt_P" + str(subj) + ".npy"
classes = pd.read_csv(data_directory + "/categories.csv")

gts = np.load(gt_path)
eye = np.load(eye_path)

# correct for images
gts, eye = corrector(gts, eye)

# select only trained labels
gts_labels = gts[~np.isin(gts[:, 0], labels_train)]
eye_labels = eye[~np.isin(gts[:, 0], labels_train)]

labels_test = np.unique(gts_labels[:, 0])

# get only trials that were used for training
eye_test = eye_labels
gts_test = gts_labels

print(gts_test.shape)
print(eye_test.shape)

# draw random indeces for original and true image
np.random.seed(42)
n_trials = 5000
all_idx = np.arange(0, len(eye_test))
idx = np.random.choice(all_idx, n_trials, replace=False)

# create dict
trial_dict = {"eye_data": [], "label_true": [], "label_false": [], "image_true": [], "image_false": [], "condition": []}
trial_dict["eye_data"] = eye_test[idx]
trial_dict["label_true"] = gts_test[idx, 0]
trial_dict["image_true"] = gts_test[idx, 1]

# draw random condition
cond1 = np.zeros(n_trials//2)
cond2 = np.ones(n_trials//2)
cond = np.concatenate([cond1, cond2])
np.random.shuffle(cond)
# put in dict
trial_dict["condition"] = cond

# where condition is 1, make false label different from true label
false_labels = np.random.choice(labels_test, n_trials, replace=True)
trial_dict["label_false"] = false_labels
# but what if we accidentally drew the true label? then draw again
mask = trial_dict["label_false"] == trial_dict["label_true"]
while mask.any():
    trial_dict["label_false"][mask] = np.random.choice(labels_test, mask.sum(), replace=True)
    mask = trial_dict["label_false"] == trial_dict["label_true"]

# where condition is 0, make false label same as true label
mask = trial_dict["condition"] == 0
trial_dict["label_false"][mask] = trial_dict["label_true"][mask]

# where both are the same
mask = trial_dict["label_false"] == trial_dict["label_true"]
# count how many are the same
if mask.sum() != n_trials//2:
    raise ValueError("Something went wrong with the conditions.")

# get images for the false condition
for trial in tqdm(range(n_trials)):
    if trial_dict["condition"][trial] == 1:
        # get images from the different label
        gt_scope = gts_test[gts_test[:, 0] == trial_dict["label_false"][trial], :]
        trial_dict["image_false"].append(gt_scope[np.random.choice(len(gt_scope), 1)[0], 1])
    else:
        # get different images from the same label
        gt_scope = gts_test[gts_test[:, 0] == trial_dict["label_true"][trial],:]
        mask = gt_scope[:, 1] != trial_dict["image_true"][trial]
        gt_scope = gt_scope[mask]
        trial_dict["image_false"].append(gt_scope[np.random.choice(len(gt_scope), 1)[0], 1])

# save trials
with open(path + "/eval_test.pkl", "wb") as f:
    pickle.dump(trial_dict, f)