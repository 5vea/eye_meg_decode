"""
Here we are now taking our previously prepared datasets and merging them into one per person.
Then we are shaping it - we will save two formats: one with the shape (n_samples, n_features, n_timepoints_trial) and one with
the shape (n_samples, n_features, n_timepoints_bin) with one file per bin.

Also, this code drops the last timepoint of each trial, to have even bins.

Furthermore, epochs where the participant blinked are dropped conditionally -- if of one trial more than 50% of the data is missing, the whole trial is dropped.

For the bin wise analysis, the same thing is done, but the criterion is applied to the bins.
"""

exclude_blink = True # if false, make zero where blink
criterium = 0.5 # criterium for complete epoch exclusion

import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

folder = r"/home/saskia_fohs/mbb_dl_project/preprocessed"

for part in range(4):
    # read the cleaned data
    part_files = glob.glob(os.path.join(folder, f"*preprocessed_binned_P{part+1}_*"))

    df_all = pd.DataFrame()
    df_all_bins = pd.DataFrame()

    for file in tqdm(part_files):
        df = pd.read_csv(file)

        # get max value of time
        max_time = df["time"].max()
        # drop where time is the max value
        df = df[df["time"] != max_time]

        # for extra bin df
        df_bins = df.copy()

        # only include epochs where there are 1680 values in x, y and p
        grouped_epochs_x = df.groupby("epoch")["x"].count().to_frame(name="count")["count"] < (1680 * criterium)
        grouped_epochs_y = df.groupby("epoch")["y"].count().to_frame(name="count")["count"] < (1680 * criterium)
        grouped_epochs_p = df.groupby("epoch")["p"].count().to_frame(name="count")["count"] < (1680 * criterium)
        grouped_epochs = grouped_epochs_x & grouped_epochs_y & grouped_epochs_p
        epochs_to_drop = grouped_epochs[grouped_epochs].index

        if exclude_blink:
            df = df[~df["epoch"].isin(epochs_to_drop)]

        # bin wise rejection -- each bin should be 120 values, so 50% missing data is 60 values
        df_bins["bin_epoch"] = df_bins["epoch"].astype(str) + "_" + df_bins["bin"].astype(str)
        grouped_bins_x = df_bins.groupby("bin_epoch")["x"].count().to_frame(name="count")["count"] < (120 * criterium)
        grouped_bins_y = df_bins.groupby("bin_epoch")["y"].count().to_frame(name="count")["count"] < (120 * criterium)
        grouped_bins_p = df_bins.groupby("bin_epoch")["p"].count().to_frame(name="count")["count"] < (120 * criterium)
        grouped_bins = grouped_bins_x & grouped_bins_y & grouped_bins_p
        bins_to_drop = grouped_bins[grouped_bins].index

        if exclude_blink:
            df_bins = df_bins[~df_bins["bin_epoch"].isin(bins_to_drop)]

        # set nan values in x, y and p to zero
        df["x"] = df["x"].fillna(0)
        df["y"] = df["y"].fillna(0)
        df["p"] = df["p"].fillna(0)

        df_bins["x"] = df_bins["x"].fillna(0)
        df_bins["y"] = df_bins["y"].fillna(0)
        df_bins["p"] = df_bins["p"].fillna(0)

        df_all = pd.concat([df_all, df])
        df_all_bins = pd.concat([df_all_bins, df_bins])

    # now lets make the numpy magic happen
    x_trial_data = df_all["x"].to_numpy().reshape(-1, 1680)[:, np.newaxis, :]
    y_trial_data = df_all["y"].to_numpy().reshape(-1, 1680)[:, np.newaxis, :]
    p_trial_data = df_all["p"].to_numpy().reshape(-1, 1680)[:, np.newaxis, :]

    gt_trial_cat = df_all["things_category_nr"].to_numpy().reshape(-1, 1680)[:, np.newaxis, :]
    gt_trial_image = df_all["things_image_nr"].to_numpy().reshape(-1, 1680)[:, np.newaxis, :]

    # stack data
    trial_data = np.concatenate([x_trial_data, y_trial_data, p_trial_data], axis=1)
    #print(trial_data.shape)
    gt_data = np.stack([gt_trial_cat, gt_trial_image], axis=1)
    gt_data = gt_data.squeeze()
    #print(gt_data.shape)

    """print("_____ trial data")
    print(gt_data[0, 0, :])
    print(gt_data[0, 1, :])"""

    # gt data can be shortened, because we only have on gt per trial/bin
    gt_data = gt_data[:, :, 0]

    # save
    np.save(os.path.join(folder, f"trial_data_P{part+1}"), trial_data)
    np.save(os.path.join(folder, f"trial_gt_P{part+1}"), gt_data)

    # save bin wise, because of bin drop, it is otherwise unclear what corresponds to what
    for b in range(13):
        x_bin_data = df_all_bins[df_all_bins["bin"] == b]["x"].to_numpy().reshape(-1, 120)[:, np.newaxis, :]
        y_bin_data = df_all_bins[df_all_bins["bin"] == b]["y"].to_numpy().reshape(-1, 120)[:, np.newaxis, :]
        p_bin_data = df_all_bins[df_all_bins["bin"] == b]["p"].to_numpy().reshape(-1, 120)[:, np.newaxis, :]

        gt_bin_cat = df_all_bins[df_all_bins["bin"] == b]["things_category_nr"].to_numpy().reshape(-1, 120)[:, np.newaxis, :]
        gt_bin_image = df_all_bins[df_all_bins["bin"] == b]["things_image_nr"].to_numpy().reshape(-1, 120)[:, np.newaxis, :]

        # stack data
        bin_data = np.concatenate([x_bin_data, y_bin_data, p_bin_data], axis=1)
        gt_data_bin = np.stack([gt_bin_cat, gt_bin_image], axis=1)
        gt_data_bin = gt_data_bin.squeeze()

        """print("_____ bin data")
        print(gt_data_bin[0, 0, :])
        print(gt_data_bin[0, 1, :])"""

        gt_data_bin = gt_data_bin[:, :, 0]

        # save
        np.save(os.path.join(folder, f"bin_data_P{part+1}_B{b}"), bin_data)
        np.save(os.path.join(folder, f"bin_gt_P{part+1}_B{b}"), gt_data_bin)