# Decode Complex Stimuli from Eye Movements to Later Check for Induced Artefacts in M/EEG Decoding

## Welcome!
The goal of this repo is to develop a method to reliably decode information from eye movements. 
For this we will use the things dataset and use a contrastive learning approach. When later incorporating M/EEG data, a 3D clip will be possible, to observe, if not only images can be decoded from eye movements, but if eye movements can also encode M/EEG data.
This project will focus on the eye movements side of things.

## Plan:
First we need an embedding model fit for eye movements. We will compare a CNN (a 1D ResNet) and a transformer encoder. Both will be trained as classifiers for 1/3 of the concepts, to mitigate later double dipping. 
The performance of the decoders is tested on one sample that was left out from all of the 1/3 training classes. This elicited that the transformer was heavily overfitting (100% Acc@1 in train and 0.32% in test), which is why random epoch wise data augmentation was introduced. The data augmentation is discussed below.

Then both will be used as enmbedding models for eye movements in seperate instances.
The non-linear embedding layers for a pretrained vision encoder (e.g. ViT-H) will be trained on the 1/3 of classes that were already trained in the classfication (the double dipping part) and 1/3 of the left out classes. Then the model will be tested in zero shot evaluation within and between classes (e.g. 2/3AFC) on the last left out 1/3 of classes.


## Data Augmentation
We tried smooth time masking as a way, to break heavy trial dependent classifications. The length of the mask is 0.1 of signal length. 
Furthermore, we added gaussian noise to the data.

Those steps were only done in training and can be found in the transformer module.
