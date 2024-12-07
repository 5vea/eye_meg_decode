"""
This code will read the simulated data and also bin the data into 50 ms bins.

Flattens in the end, because we want to explore the data without bin systematics.

Assumes fs = 1000 Hz.
"""

import numpy as np

bin_size = 50 # ms
flatten = True

# load data
gt_trials = np.load("../simulate_eye/gt_trials.npy")
trials = np.load("../simulate_eye/trials.npy")

# bin data
# make to size (n, n_bins, bin_size)
n_bins = trials.shape[2] // bin_size

binned_trials = np.zeros((trials.shape[0], n_bins, 3, bin_size))

# repmat gt to n_bins
gt_trials = np.repeat(gt_trials[:,np.newaxis], n_bins, axis=1)

for i in range(n_bins):
    binned_trials[:, i, :, :] = trials[:, :, i*bin_size:(i+1)*bin_size]

# check if data format should be flattened before output (no bin systematics)
if flatten:
    binned_trials = binned_trials.reshape(-1, binned_trials.shape[2], binned_trials.shape[3])
    gt_trials = gt_trials.flatten()

print("Size GT", gt_trials.shape)
print("Size Trials", binned_trials.shape)

# save data
np.save("../data/data_eye/gt_trials_preprocessed", gt_trials, allow_pickle=True)
np.save("../data/data_eye/trials_preprocessed", binned_trials)