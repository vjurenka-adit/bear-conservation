# Masterclasses

Date: 2024-01-29

## HackThePlanet

Detecting Bears on Edge on a very small microcontroller.
Collaboration to protect bears and humans - NPO in Romania.
Need to prevent bear-conflicts at all times.
Trophy hunting were a thing in Romania.

Speaker: Thijs Suijten

4G camera traps - take pictures - the model analyzes the data in the cloud
Lights and Speakers are turned on
The whole round trip takes 20s-30s (from taking the picture to scaring the bear)

Last line of defense in the villages. Analyze a video feed to instantly
trigger a repeller.

technical blog post for the project: https://engineering.q42.nl/ai-bear-repeller/

### Cost of a false positive?

- Battery drainage?

### Dataset

#### Summary

- 15k bears
- 15k other animals (lots of wild dogs)
- 3k blanks

#### Format

- images and videos
- extracted frames from videos
- subset of a larger dataset
- the folder structure contains location information (please ignore)

### Current model & Challenges

- Image classification did not work
  - The ML model was learning the background of the image (eg. Pool in the background)
- Lots of repetition in some of the sites
- Switched to object detection
- Current model YOLOv5 small
  - Use the MD framework

__Idea__: Background subtraction preprocessing step

### Hardware

Most important thing: reliable model on platform that we can attach a camera to:

- IMX93 (Linux based)
- RT1062 (OpenMV/Edge Impulse to export the model for the microcontroller)

The system needs to work both on day and night - there is a dataset for thermal vision.

### Partnership

https://www.carpathia.org/

## BearID

Speaker: Melanie Clapham and Ed Miller

Non Profit Research Organisation: `BearID Project`
Develop non invasive tools to monitor, conserve and study bears.

Develop a robust, open-souce automated software tool to identify
individual bears with > 90% accuracy.

Current photo-ID performances:

- Detection: 0.98
- Identification: 0.84

### How can re-ID benefit bear conservation

- population monitoring
- land management
- behaviour research
- human wildlife conflict and coexistence

### British Columbia

Highest density of grizzly bears in the world.
Temperate rainforest ecosystems.
Feed in the coastline too.
Rely on Pacific salmon fishes.
Good place to research.
Able to record individual ids, age, sex, behaviour and interactions between individuals.
Camera traps to detect bears and other animals and check them once a month.

### Partnership

https://conservationxlabs.com to run ML models.

### Be aware of

- Bears coat varies time of the year and when wet/dry, different lightning
- Some bears can look very similar (familial ties)
- Robustness is more important than highest accuracy
- Confidence in the model is important (ie predictability)
if different models for different seasons work better, thats ok.
- Face chips may need augmentation (saturation, exposure), to imitate camera trap images.
- If models are too large for edge devices that's ok!
- Bear Ears have up to 6-7 positions

### Avoid data leakage using different splits

Be very cautious as it will overinflate the performance.

### Technical presentation

Senior Principal Engineer at Arm
Co-director, BearID
AWS Machine Learning Hero

#### FaceNet

Face detection -> face reorientation -> face encoding -> face matching
Object detection model + landmark detection to do the face reorientation.

#### dlib: Dog Hipsterizer

Worked well with dogs and bears.
Used the tool to build the dataset.

### Datasets

Data Limitation: Closed dataset
How do we consider new individuals?

#### brooksFalls

__brooksFalls__: 3372 images - 80/20 split train/test
two xml files (test and train datafiles)

bounding box information + labels + landmarks

A lot of Individuals with a few images
Individuals with a lot of images.

Pose variation issues (forward, down/right, left, down)

#### Chip Dataset

150x150 pixel crop
4669 chips
68 / 12 / 20 split

xml files for the chips are similar from dlib

### Face encoding

Take pairs of images and pairs of different individuals
Test protocol: _Labeled Faces in the Wild_

132 individuals, 3740 training images, 934 test images
5-fold cross validation

### Classification

Use SVM from the encodings:
- prone to overfitting
- does

Probe-Gallery using distant metrics
- Infer embedding for test images
- Compare with training image embeddings (gallery)
- Find closest match using euclidean distance
  - Top-1 Rank
  - Top-5 Rank
  - K-Nearest Neighbors

Idea: Make the splits such that none of the individuals are in the training set

## TODO

- Read the two papers from BearID
- Read the engineering blog post from HackThePlanet: https://engineering.q42.nl/ai-bear-repeller/
- Think about how to avoid data leakage
