"""
This code matches the categories: things put as "categories" in the eye tracking data to their labels.
It will return a complete csv with category-number, image-number, label and image id.
Also, this can then be used to adapt the dataloader and the training algorithm to the right amount of classes.
"""

import pandas as pd
import glob
import os
from tqdm import tqdm

folder = r"/home/saskia_fohs/mbb_dl_project/preprocessed"

#read all original eye data files
all_files = glob.glob(os.path.join(folder, "*cleaned_P*.csv"))

# empty frame to concat to
all_df = pd.DataFrame()

for file in tqdm(all_files):
    df = pd.read_csv(file)

    # only keep things_image_nr, things_category_nr and image_path
    df = df[["things_image_nr", "things_category_nr", "image_path"]]

    df["category"] = df["image_path"].apply(lambda x: x.split("/")[-2])
    df["image_id"] = df["image_path"].apply(lambda x: x.split("/")[-1].split(".")[0])

    # group by things_image_nr
    df = df.groupby("things_image_nr").first().reset_index()

    all_df = pd.concat([all_df, df])

# group by things_image_nr
all_df = all_df.groupby("things_image_nr").first().reset_index()

print(all_df.head())
print(len(all_df))

all_df.to_csv(os.path.join(folder, "categories.csv"), index=False)