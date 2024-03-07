# bearidentification

In this section, we describe how to train and run the
bearidentification model.

![Identification of a bear](./assets/images/model_output/identification/prediction_at_5_individuals_5_samples_per_individual.png)

## Data

### Chips

Run the following command to install the chips:

```sh
make bearidentification_data_install_chips
```

If you are curious and would like to know how they are generated, check out the
[documention for bearfacesegmentation](./bearfacesegmentation.md)

### Datasplits

We decided to perform two distinct types of datasplits to evaluate our models
for different use cases.
1. __Splitting by individuals__: It allows to  assess how good the model is at
   identifying and clustering unseen individuals.
2. __Splitting on the provided BearID split__: The BearID team provided a
   curated split where individuals are spread across the different
train/val/test splits, avoiding data leaks of images taken the same day for the
same individual (bursts of images for instance).

To generate these datasplits, run the following command:

```sh
make bearidentification_data_split
```

## Training

A Metric Learning approach is taken to learn a good embedding space for
bear faces.

To train all different models and experiments, use the following command:

```sh
make bearidentification_metriclearning_train
```

To train the best models, run the following command:

```sh
make bearidentification_metriclearning_train_best
```

To train the baselines, run the following command:

```sh
make bearidentification_metriclearning_train_baselines
```

## Evaluation

To aggregate the performances of the diffent trained models in a csv
file, one can run the following command:

```sh
make bearidentification_metriclearning_eval_summary
```

## Packaging the pipeline

Once the bearfacesegmentation model and the bearidentification model are
trained, one can package the end to end pipeline to make it easy to ship
it around and to make predictions.

```sh
make package_pipeline
```

If you downloaded a packaged pipeline archive (usually a
`packaged_pipeline.zip` file) you can install it with the following
command:

```sh
make install_packaged_pipeline
```

We are now ready to start making predictions with the pipeline.

## Prediction

Run the following command to make a prediction on the provided bear image.

```sh
make identify_default
```

## Fast Track

To run the datasplit, the training and the evaluation at once, run the
following command:

```sh
make bearidentification_metriclearning
```
