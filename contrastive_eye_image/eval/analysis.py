import pickle
import numpy as np
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt

# load data
subjs = [1, 2]
models = ["transformer", "cnn"]
evals = ["train", "test"]

# new dataframe
df = pd.DataFrame(columns=["subj", "model", "eval", "acc_cond0", "acc_cond1", "acc", "diff_cond0", "diff_cond1", "diff"])

for subj in subjs:
    for model in models:
        for eval in evals:
            path = f"/home/saskia_fohs/mbb_dl_project/models/subj{subj}/{model}_CL_2_1000_0-0003_50"

            print(f"Subject: {subj}, Model: {model}, Eval: {eval}")

            # load data
            if eval == "test":
                test_path = path + "/eval_test.pkl"
                with open(test_path, 'rb') as f:
                    eval_data = pickle.load(f)
            else:
                test_path = path + "/eval_train.pkl"
                with open(test_path, 'rb') as f:
                    eval_data = pickle.load(f)

            corr_mask = np.array(eval_data["val_true"]) > np.array(eval_data["val_false"])
            # only keep where condition 1
            acc_cond0 = np.mean(corr_mask[eval_data["condition"] == 0])
            print(f"Accuracy condition 0: {acc_cond0}")
            # only keep where condition 2
            acc_cond1 = np.mean(corr_mask[eval_data["condition"] == 1])
            print(f"Accuracy condition 1: {acc_cond1}")
            # overall accuracy
            acc = np.mean(corr_mask)
            print(f"Accuracy: {acc}")

            eval_data["diff"] = np.array(eval_data["val_true"]) - np.array(eval_data["val_false"])
            # for cond 0
            diff_cond0 = np.mean(eval_data["diff"][eval_data["condition"] == 0])
            print(f"Diff condition 0: {diff_cond0}")
            # for cond 1
            diff_cond1 = np.mean(eval_data["diff"][eval_data["condition"] == 1])
            print(f"Diff condition 1: {diff_cond1}")
            # overall diff
            diff = np.mean(eval_data["diff"])
            print(f"Diff: {diff}")

            diff_cond0_corr = np.mean(eval_data["diff"][eval_data["condition"] == 0][corr_mask[eval_data["condition"] == 0]])
            print(f"Diff condition 0 correct: {diff_cond0_corr}")
            diff_cond1_corr = np.mean(eval_data["diff"][eval_data["condition"] == 1][corr_mask[eval_data["condition"] == 1]])
            print(f"Diff condition 1 correct: {diff_cond1_corr}")

            diff_cond0_incorr = np.abs(np.mean(eval_data["diff"][eval_data["condition"] == 0][~corr_mask[eval_data["condition"] == 0]]))
            print(f"Diff condition 0 incorrect: {diff_cond0_incorr}")
            diff_cond1_incorr = np.abs(np.mean(eval_data["diff"][eval_data["condition"] == 1][~corr_mask[eval_data["condition"] == 1]]))
            print(f"Diff condition 1 incorrect: {diff_cond1_incorr}")

            # concat to dataframe
            df = pd.concat([df, pd.DataFrame({"subj": [subj], "model": [model], "eval": [eval], "acc_cond0": [acc_cond0], "acc_cond1": [acc_cond1], "acc": [acc], "diff_cond0": [diff_cond0], "diff_cond1": [diff_cond1], "diff": [diff], "diff_cond0_corr": [diff_cond0_corr], "diff_cond1_corr": [diff_cond1_corr], "diff_cond0_incorr": [diff_cond0_incorr], "diff_cond1_incorr": [diff_cond1_incorr]})])

offset=[-0.03, 0.03]
# save figure
fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
for i, model in enumerate(models):
    for j, eval in enumerate(evals):
        data = df[(df["model"] == model) & (df["eval"] == eval)]
        ax[i].bar(data["subj"]+offset[j], data["acc"], label=eval, alpha=1)
        ax[i].set_title(f"Eye Encoder: {model}")
        ax[i].set_xlabel("Subject")
    if i == 0:
        ax[i].set_ylabel("Accuracy")
    ax[i].legend()
    ax[i].set_xticks(subjs)
    ax[i].set_xticklabels(subjs)
    ax[i].set_yticks(np.arange(0.5, 1.05, 0.1))
    ax[i].set_ylim([0.45, 1])
    ax[i].axhline(y=0.5, color='k', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("acc_plot.png")

# get rid of train column
df = df[df["eval"] == "test"]

# plot the two conditions comparative for each model with a bar resembling the mean accuracy of both subjects
fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
for i, model in enumerate(models):
    data = df[df["model"] == model]
    ax[i].bar(data["subj"]+offset[0], data["acc_cond0"], label="Within Label", alpha=0.5)
    ax[i].bar(data["subj"]+offset[1], data["acc_cond1"], label="Between Label", alpha=0.5)
    ax[i].set_title(f"Eye Encoder: {model}")
    ax[i].set_xlabel("Subject")
    ax[i].legend()
    ax[i].set_xticks(subjs)
    ax[i].set_xticklabels(subjs)
    ax[i].set_yticks(np.arange(0.5, 1.05, 0.1))
    ax[i].set_ylim([0.45, 1])
    ax[i].axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
    if i == 0:
        ax[i].set_ylabel("Accuracy")

plt.tight_layout()
plt.savefig("acc_cond_plot.png")

# plot the two conditions comparative for each model with a bar resembling the mean accuracy of both subjects
fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
for i, model in enumerate(models):
    data = df[df["model"] == model]
    ax[i].bar(data["subj"]+offset[0], data["diff_cond0_corr"], label="Within Label", alpha=0.5)
    ax[i].bar(data["subj"]+offset[1], data["diff_cond1_corr"], label="Between Label", alpha=0.5)
    ax[i].set_title(f"Eye Encoder: {model}")
    ax[i].set_xlabel("Subject")
    ax[i].legend()
    ax[i].set_xticks(subjs)
    ax[i].set_xticklabels(subjs)
    ax[i].set_yticks(np.arange(0.5, 1.05, 0.1))
    ax[i].set_ylim([0.45, 1])
    ax[i].axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
    if i == 0:
        ax[i].set_ylabel("Confidence")

plt.tight_layout()
plt.savefig("diff_cond_plot.png")

fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
for i, model in enumerate(models):
    data = df[df["model"] == model]
    ax[i].bar(data["subj"]+offset[0], data["diff_cond0_incorr"], label="Within Label", alpha=0.5)
    ax[i].bar(data["subj"]+offset[1], data["diff_cond1_incorr"], label="Between Label", alpha=0.5)
    ax[i].set_title(f"Eye Encoder: {model}")
    ax[i].set_xlabel("Subject")
    ax[i].legend()
    ax[i].set_xticks(subjs)
    ax[i].set_xticklabels(subjs)
    ax[i].set_yticks(np.arange(0.5, 1.05, 0.1))
    ax[i].set_ylim([0.45, 1])
    ax[i].axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
    if i == 0:
        ax[i].set_ylabel("Confidence")

plt.tight_layout()
plt.savefig("diff_cond_plot_incorr.png")