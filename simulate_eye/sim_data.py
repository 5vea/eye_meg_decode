"""
Here we will simulate some data to test the functions in the other files.
This serves to be able to work on the project, until we have settled on a dataset to use.

In the current state it will only return random data, but if we cannot settle on a dataset, we can extend this to
simulate systematic differences between the classes.

Assumes fs = 1000 Hz.
"""

import numpy as np

def simulate_trials(n=1000, gts=[1,2,3,4], ts=10000, v=False):
    """
    Simulate n trials, with timepoints ts, and possible ground truths gts.
    """
    gt_trials = np.random.choice(gts, n)
    trials = np.random.randn(n, 3, ts)

    if v:
        print("Size GT", gt_trials.shape)
        print("Size Trials", trials.shape)

    return gt_trials, trials

# simulate data, from gabor patches with different orientations
# 0° = 180°, 45° = 225°, 90° = 270°, 135° = 315° --> this is why we stop at 135°
gt_trials, trials = simulate_trials(n=1000, gts=[0, 45, 90, 135], ts=2000, v=True)

# save data
np.save("gt_trials.npy", gt_trials, allow_pickle=True)
np.save("trials.npy", trials)