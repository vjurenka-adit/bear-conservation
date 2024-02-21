# Automated Facial Recognition for wildlife that lack unique markings: A DL approach for brown bears

## Open Source application

- [Github Repository](https://github.com/hypraptive/bearid)
- [Pretrained models](https://github.com/hypraptive/bearid-models)
- [Archive](https://doi.org/10.5281/zenodo.4014233)

## Authors

- [Melanie Clapham](https://orcid.org/0000-0001-8924-7293)
- [Chris T. Darimont](https://orcid.org/0000-0002-2799-6894)

## Challenges

Unique challenges with bears:

- Vary in morphology across their range
- Experience extreme weight fluctuations between seasons as they age and grow

## Application pipeline

```txt
Face detection -> Face reorientation and cropping -> Face encoding -> Face classification
```

## Data Collection

- Knight Inlet, British Columbia
- Brooks River, Katmai National Park, Alaska

Images taken by DSLK cameras (various models and focal lenghts)
Resolution ranges from 0.3MB to 24.1MB

4675 images of 132 individuals
median = 22 images/individuals [range 1-242] with visible faces - criteria: both eyes visible

## Training and test data

Golden dataset from the 4675 images: include bounding box for each face, the locations of the landmarks and identification of each bear.

imglab can be used to review the annotations.

### Split

Random split the golden dataset into 80/20 split.

### Image preprocessing

#### Face Detection

all training images are down to 200x1500 pixels

If a face becomes to small (200x200 pixels) they scaled until the face was 200x200 and then cropped the overall image to 2000x1500

## Face Detection - bearface

Object Detector (OD) and Shape predictor (SP)

### Object Detector

Sliding window + CNN trained with dlibs max margin.

Trained using the bbox labels from the golden dataset.

### Shape Predictor

Trained using the landmark labels in the golden dataset.

## Face reorientation and cropping - bearchip

It uses the facial landmarks created by bearface to rotate and center the face. 
Only uses the eyes to align and center images.
Scale and crop (150x150 pixels) the faces and writes each of them as a JPEG.

## Face encoding - bearembed

Leverage a similarity metric to learn a function that maps an input image into a target space.

The output is an embedding of a facial image that can be compared to other embeddings to identify individuals using a face classifier.

Similarity comparison network using a CNN with a ResNet34 backbone.
The embedding is a 128 dimensional vector.

Metric Learning was done using a pairwise hinge loss rather than a triplet loss.
Hard negative mining is used to ensure a balanced ratio with mini-batches of positive and negative pairs.

## Face classification - bearsvm

Assign an individual ID label to an embedding created by bearembed.
