"""
Zero Shot with pairwise comparison

We will use the otherwise generated dataset of eye data with image pairs.
For each participant, there is an evaluation file in their respective model folder.
There are 2 participants with two models each. The models are a transformer and a CNN model.

"""

path = "/home/saskia_fohs/mbb_dl_project/models/subj1/cnn_CL_2_1000_0-0003_50"
eval = "test"

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import numpy as np
import argparse
import json
import pandas as pd
import torchvision
from PIL import Image
import itertools
import sys
sys.path.insert(0, '..')
import transformer_cl_model as tf
import cnn_cl_model as cnn
import pickle
from tqdm import tqdm

# load checkpoint
checkpoint_path = os.path.join(path, 'checkpoint.pth')
print(checkpoint_path)
checkpoint = torch.load(checkpoint_path, weights_only=True)

# create model
if "cnn" in path:
    model = cnn.ContrastiveModel_cnn()
elif "transformer" in path:
    model = tf.ContrastiveModel_tf()
else:
    raise ValueError("Model not found.")

model.load_state_dict(checkpoint['state_dict'])
model.eval()

# load data
if eval == "test":
    test_path = path + "/eval_test.pkl"
    with open(test_path, 'rb') as f:
        eval_data = pickle.load(f)
else:
    test_path = path + "/eval_train.pkl"
    with open(test_path, 'rb') as f:
        eval_data = pickle.load(f)

# classes csv
data_directory = "/home/saskia_fohs/mbb_dl_project/preprocessed"
classes = pd.read_csv(data_directory + "/categories.csv")
img_dir = "/home/saskia_fohs/mbb_dl_project/things_images"

# model to cuda
model = model.to("cuda")

# matching between eye movement and images
# similar to https://github.com/mlfoundations/open_clip
def match_eye_image(eye, image_corr, image_false):
    """
    Matches the eye movement data to the images.
    """

    images = {"corr": image_corr, "false": image_false}

    # preprocess image
    for key, value in images.items():
        images[key] = Image.open(img_dir + "/" + classes[classes["things_image_nr"] == value]["image_path"].values[0])
        images[key] = torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1.transforms()(images[key]).unsqueeze(0)
    # stack images
    images = torch.cat([images["corr"], images["false"]], dim=0)

    # preprocess eye
    eye = torch.from_numpy(eye).float().unsqueeze(0)

    with torch.no_grad():
        # get image embeddings
        images = images.to("cuda")
        image_features = model.image_encoder(images)
        image_embeddings = model.projection_image(image_features)

        # get eye embeddings
        eye = eye.to("cuda")
        eye_embeddings = model.projection_eye(model.eye_encoder(eye))

        # normalize vectors # CAVEAT: NOT DONE IN TRAINING
        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
        eye_embeddings /= eye_embeddings.norm(dim=-1, keepdim=True)

        image_probs = torch.softmax(100 * eye_embeddings @ image_embeddings.T, dim=-1)

        return image_probs[0][0], image_probs[0][1]

val_corr = []
val_false = []
for i in tqdm(range(len(eval_data["eye_data"]))):
    #print(f"trial {i}")
    prob_corr, prob_false = match_eye_image(eval_data["eye_data"][i], eval_data["image_true"][i], eval_data["image_false"][i])
    val_corr.append(prob_corr.detach().cpu().numpy())
    val_false.append(prob_false.detach().cpu().numpy())

    """if i == 100:
        break"""

eval_data["val_true"] = val_corr
eval_data["val_false"] = val_false

# save
with open(test_path, 'wb') as f:
    pickle.dump(eval_data, f)

corr_mask = np.array(val_corr) > np.array(val_false)
# only keep where condition 1
acc_cond0 = np.mean(corr_mask[eval_data["condition"][:len(val_corr)] == 0])
print(f"Accuracy condition 0: {acc_cond0}")
# only keep where condition 2
acc_cond1 = np.mean(corr_mask[eval_data["condition"][:len(val_corr)] == 1])
print(f"Accuracy condition 1: {acc_cond1}")
# overall accuracy
acc = np.mean(np.array(val_corr) > np.array(val_false))
print(f"Accuracy: {acc}")