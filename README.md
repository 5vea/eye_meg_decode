# Decode Complex Stimuli from Eye Movements to Later Check for Induced Artefacts in M/EEG Decoding

## Welcome!
The goal of this repo is to develop a method to reliably decode information from eye movements. 
For this we will use the things dataset and use a contrastive learning approach. When later incorporating M/EEG data, a 3D clip will be possible, to observe, if not only images can be decoded from eye movements, but if eye movements can also encode M/EEG data.
This project will focus on the eye movements side of things.

## Plan:
First we need an embedding model fit for eye movements. We will compare a CNN (a 1D ResNet) and a transformer encoder. Both will be trained as classifiers for 1/3 of the concepts, to mitigate later double dipping. 
There performance as decoders can then be tested on the 2/3 of held out classes.

Then both will be used as enmbedding models for eye movements in seperate instances.
The non-linear embedding layers for a pretrained vision encoder (e.g. ViT-H) will be trained on the 1/3 of classes that were already trained in the classfication (the double dipping part) and 1/3 of the left out classes. Then the model will be tested in zero shot evaluation within and between classes (e.g. 2/3AFC) on the last left out 1/3 of classes.

## Things we may want to do:
Not use the whole class in training, but also keep class samples for testing.
