"""
This code iterates through all participants and sessions and applies a baseline correction to the eye data. The baseline correction is done by subtracting the mean of the x, y, and p values before timepoint 0 from the respective values at each timepoint. The corrected data is then binned into 13 bins based on the timepoints.
Also, a bin index is applied, so that the data can be split into bins for later fine grained analysis.

Also test images are deleted, because those were either catch images or repeated images.
"""

import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

folder = r"/home/saskia_fohs/mbb_dl_project/preprocessed"

timepoints = np.arange(-100, 1301, 1)
timepoints_starts = np.arange(-100, 1201, 100)
timepoints_ends = np.arange(0, 1301, 100)
bins = np.arange(0, 14, 1)

print(timepoints_starts)
print(timepoints_ends)
print(bins)

# mapping function for timepoints to bins
def map_time_to_bin(timepoint):
    starts = timepoints_starts <= timepoint
    ends = timepoints_ends > timepoint
    return bins[starts & ends]

for part in range(4):
    part_files = glob.glob(os.path.join(folder, f"*cleaned_P{part+1}_*.csv"))
    for session, file in tqdm(enumerate(part_files)):
        df = pd.read_csv(file)
        # rename columns UADC009-2104,UADC010-2104,UADC013-2104 to x, y and p
        df = df.rename(columns={"UADC009-2104": "x", "UADC010-2104": "y", "UADC013-2104": "p"})

        # baseline correction -- maybe not necessary and not the best way to do it [! work on it later]
        for epoch in tqdm(df["epoch"].unique()):
            epoch_df = df[df["epoch"] == epoch]
            baseline = epoch_df[epoch_df["time"] < 0]["x"].mean()
            df.loc[df["epoch"] == epoch, "x"] = df.loc[df["epoch"] == epoch, "x"] - baseline
            baseline = epoch_df[epoch_df["time"] < 0]["y"].mean()
            df.loc[df["epoch"] == epoch, "y"] = df.loc[df["epoch"] == epoch, "y"] - baseline
            baseline = epoch_df[epoch_df["time"] < 0]["p"].mean()
            df.loc[df["epoch"] == epoch, "p"] = df.loc[df["epoch"] == epoch, "p"] - baseline

        # apply mapping function
        df["bin"] = df["time"].apply(map_time_to_bin)

        # drop where test_image_nr is not empty
        df = df[df["test_image_nr"].isnull()]

        # save to csv -- SESSION IS BEING REORDERED BY THIS -- DOES NOT MATTER
        df.to_csv(os.path.join(folder,f"preprocessed_binned_P{part+1}_S{session+1}"), index=False)