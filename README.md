# Decode Complex Stimuli from Eye Movements to Later Check for Induced Artefacts in M/EEG Decoding

## Welcome!
The goal of this repo is to develop a method to reliably decode information from eye movements. 
For this we will use the things dataset and use a contrastive learning approach. When later incorporating M/EEG data, a 3D clip will be possible, to observe, if not only images can be decoded from eye movements, but if eye movements can also encode M/EEG data.
This project will focus on the eye movements side of things.

## Plan and Goal
Please refer to our little goup paper in the global folder: paper_kürten-fohs.pdf

## Important Folders
### preprocess_eye
There are two important files in this folder preproc_things_format.py and preproc_eyes_bin_index_baseline.py. The latter is the first step on preparing the eye-movement data and the former transforms the preprocessed trials to a dataset friendly structure.

### preprocess_categories
This is needed to match the images to their IDs and label ground truths (match_categories.py).

### decode_eye
Here you can find the two classifier models (CNN_model.py and transformer_model.py) and their training script (train_model.py).

### contrastive_eye_image
In this folder are the model scripts (transformer_cl_model.py and cnn_cl_model.py) and their training script (train_cl_model.py) for the unsupervised learning approach. Also the evaluation is in a subfolder here.
#### eval
In this subfolder, you can find the task creations to test our models with (test_script.py and test_script_left_out.py) and the zero-shot execution (zero_shot_pairwise.py). Further there is the analysis of the zero-shot evaluation in this folder (analysis.py).

### interpretations
#### rsa
Sveas assignment.
#### attention_maps
Saskias assignment.

The other folders are not important for the current state of the project. data and simulate_eye served as testing grounds and decode_meg and preprocess_meg are place holders for neural processing.
