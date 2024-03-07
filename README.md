# AI for Bears

This folder contains data and work done during the AI for bears
challenge organized by [FruitpunchAI](https://www.fruitpunch.ai/).

## Introduction

Bears, particularly brown bears, are not only charismatic but also serve as
indicator and umbrella species. By understanding and protecting them, we
contribute to the overall health of the environment. However, monitoring bears
is difficult due to their elusive and wide-ranging nature. The toolbox of
methods available to study bears in non-invasive ways is limited, leading to a
reduced understanding of their population status and trends.

Human-bear conflicts can pose challenges to effective bear conservation and
threats to people and property. Promoting coexistence and mitigating human-bear
conflicts is key to the long-term conservation of brown bears, preserving the
vital role they play in maintaining a balanced ecosystem.

## Setup

### Virtualenv

Create a virtualenv using your tool of
choice (eg. conda, pyenv, regular python,
...) and activate it.

```sh
conda create -n fruitpunch_bears python=3.9
conda activate fruitpunch_bears
```

### Installing python dependencies

Then one can run the following command to install the python dependencies:

```sh
make setup
```

### git-lfs

Make sure [`git-lfs`](https://git-lfs.com/) is installed on your system.

Run the following command to check:

```sh
git lfs install
```

If not installed, one can install it with the following:

#### Linux

```sh
sudo apt install git-lfs
git-lfs install
```

#### Mac

```sh
brew install git-lfs
```

#### Windows

Download and run the latest [windows installer](https://github.com/git-lfs/git-lfs/releases).

## Identify bears


### Install the packaged pipeline

Run the following command to install the pipeline:

```sh
make install_packaged_pipeline
```

### Predict

Use the dummy predition script to check that everyhing works as expected:

```sh
make identify_default
```

You should be able to find the predictions in the folder
`./data/07_model_output/identify/default/`.

Now you can start predicting on your own images using the following python script:

```sh
python ./scripts/identify.py \
  --source-path ./data/09_external/identify/P1250243.JPG \
  --output-dir ./data/07_model_output/identify/default/
```

![Identification of a bear](./docs/development/assets/images/model_output/identification/prediction_at_5_individuals_5_samples_per_individual.png)

## Development

To contribute to this repository, one can follow the [relevant
documentation](./docs/development/README.md).

## Partnerships

This project was hosted and made possible by the following organizations:

- [FruitpunchAI](https://app.fruitpunch.ai/challenge/ai-for-bears)
- [BearID Project](https://bearresearch.org/)
- [HackThePlanet](https://www.hack-the-planet.io/)
