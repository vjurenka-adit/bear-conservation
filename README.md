# AI for Bears

This folder contains data and work done during the AI for bears
challenge organized by [FruitpunchAI](https://www.fruitpunch.ai/).

## Structure, organisation and conventions

The project is organized following the [cookie-cutter-datascience
guideline](https://drivendata.github.io/cookiecutter-data-science/#directory-structure).

### Code

The code is organised in different projects, namely `beardetection`,
`bearfacedetection`, `bearfacesegmentation`, `bearidentification` and
`bearedge`.

### Data

All the data lives in the `data` folder and follows some [data engineering
conventions](https://docs.kedro.org/en/stable/faq/faq.html#what-is-data-engineering-convention).
Each project should use these subfolders according to their project names (eg.
`data/05_model_input/beardetection`).

### Notebooks

The notebooks live in the `notebooks` folder and are also keyed by the project
names.

### Scripts

The scripts live in `scripts` folder and are keyed by the project names.

### Makefile

A Makefile makes it easy to prepare commands and execute them in a DAG fashion.
If we need something more involved for running code, we will add it later.

## Setup

Create a virtualenv using your tool of
choice (eg. conda, pyenv, regular python,
...) and activate it.

```sh
conda create -n fruitpunch_bears python=3.9
conda activate fruitpunch_bears
```

Then one can run the following command to install the python dependencies:

```sh
make setup
```

## Run/Write/Edit Notebooks

Run the following command to start a jupyter lab environment and start editing/running your notebooks:

```sh
make dev_noteboook
```

## Download the dataset

Install [rclone](https://rclone.org/install/) and configure a remote for
your Google Drive following this
[documentation](https://rclone.org/drive/).

```sh
make download_dataset
```

## bearfacedetection

In this section, we describe how to train the bearfacedetection object detector.

### Build the model input

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

### Training bearfacedetection

Run the following command:

```sh
make bearfacedetection_train_baseline_golden_dataset
```

_Note_: Training on CPU is possible with this setup but can take about 10/15minutes.

### Inference with bearfacedetection

Run the following command:

```sh
make bearfacedetection_train_baseline_golden_dataset
```

It should populate the `data/07_model_output` with predictions from the
finetuned model.

## bearfacesegmentation

### Download SAM checkpoint weights

Run the following command:

```sh
make download_sam_weights
```


## Tools

- [Gitlab Repositories](https://gitlab.com/groups/fruitpunch/projects/ai-for-bears)
- [Notion Board](https://www.notion.so/fruitpunch/AI-for-Bears-91a90f10263a421083a2cb075ffe53a3)
