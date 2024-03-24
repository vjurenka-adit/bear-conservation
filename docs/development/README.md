# Development

## Sub projects

### Detection

- [beardetection](./beardetection.md)

### Identification

- [bearfacedetection](./bearfacedetection.md)
- [bearfacelandmarketection](./bearfacelandmarkdetection.md)
- [bearfacesegmentation](./bearfacesegmentation.md)
- [bearidentification](./bearidentification.md)

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

## Run/Write/Edit Notebooks

Run the following command to start a jupyter lab environment and start editing/running your notebooks:

```sh
make dev_noteboook
```

## Download the datasets

### Provided datasets (HackThePlanet, BearID, SLU, ...)

Install [rclone](https://rclone.org/install/) and configure a remote for
your Google Drive following this
[documentation](https://rclone.org/drive/).

```sh
make download_dataset
```

### Roboflow bearfacedetection dataset

Find the private key on
[roboflow](https://app.roboflow.com/fruitpunch-ai-private-workspace-7nsdr/bearface-lk7vt/1).
Click on Export Dataset, select YOLOv8
format and show download code. The raw
URL is displayed and the private key is
located after the `key` parameter:
`https://app.roboflow.com/ds/b8vuUrGhDn?key=***`

One can use the following command:

```sh
PRIVATE_KEY=findmeonroboflow make download_roboflow_bearfacedetection
```

Or can export the `PRIVATE_KEY` as follows before running the subsequent commands:

```sh
export PRIVATE_KEY=findmeonroboflow
```

### All

To download all data, run the following command:

```sh
export PRIVATE_KEY=findmeonroboflow
make data
```

## Setup

### Virtualenv

Create a virtualenv using your tool of
choice (eg. conda, pyenv, regular python,
...) and activate it.

```sh
conda create -n ai4bears python=3.9
conda activate ai4bears
```

### Installing python dependencies

Then one can run the following command to install the python dependencies:

```sh
make dev_setup
```
