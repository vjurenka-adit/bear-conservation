# bearfacedetection

In this section, we describe how to train the bearfacedetection object detector.

![Detected bear face using the bear face detector](./assets/images/model_output/facedetection/image1.jpg)

## Fast Track for bearfacedetection

To download the data, prepare it, train an object detector and run
inference, one can run the following command:

```sh
make bearfacedetection
```

## Labeling

We had to relabel the provided dataset (3359 bear face images from
BearID) using Roboflow. We wanted to include the ears and mouths of the
bears.

### Roboflow Instructions

Adjust the bounding box to accurately encompass **the entire bear head**, ensuring it covers **both ears**, the **nose** and the **mouth**. 
Strive for a compact bounding box that encapsulates the complete head and fur while minimizing unnecessary space.

## Build the model input

YOLOv8 models use a very specific folder structure to be trained.

```sh
$ tree -L 2 .
.
├── data.yaml
├── train
│   ├── images
│   └── labels
└── val
    ├── images
    └── labels
```

with the data.yaml file being something like so:

```yaml
train: ./train/images
val: ./val/images
nc: 1
names:
  - bearface
```

To generate this input from the `data/01_raw` data, one should use the
following commands:

```sh
make bearfacedetection_data
```

It should populate the `data/04_feature` and `data/05_model_input` folders.

## Training bearfacedetection

### Baselines

#### Golden dataset

Run the following command:

```sh
make bearfacedetection_train_baseline_golden_dataset
```

_Note_: Training on CPU is possible with this setup but can take about 10/15minutes.

#### Roboflow relabelled dataset

Run the following command:

```sh
make bearfacedetection_bearfacedetection_train_baseline_roboflow
```

### Train all

To train all models, run the following command:

```sh
make bearfacedetection_train
```

## Inference with bearfacedetection

Run the following command:

```sh
make bearfacedetection_train_baseline_golden_dataset
```

It should populate the `data/07_model_output` with predictions from the
finetuned model.
