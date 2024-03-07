# bearfacesegmentation

In this section, we describe how to train and run the bearfacesegmentation
segmentor model.

![Segmented face of a bear](./assets/images/model_output/facesegmentation/image1.jpg)

## Generate chips

### Test the setup with test images

Some test images are provided to test that the chip generation script works properly.
The test images are located in `data/09_external/bearid/images/bears/`.

Use the following python script (change your parameters as needed):

```sh
python ./scripts/bearfacesegmentation/chips/generate.py \
  --source-dir ./data/09_external/images/bears/ \
  --save-path ./data/07_model_output/bearfacesegmentation/chips/test/ \
  --instance-segmentation-model-weights ./data/06_models/bearfacesegmentation/yolov8/roboflow_relabelled_current_best/weights/best.pt \
  --loglevel "info"
```

The following chips should be generated in your `save-path`:

![Chip 1](./assets/images/chips/image1.jpg) ![Chip 2](./assets/images/chips/image2.jpg) ![Chip 3](./assets/images/chips/image3.jpg)

### Generate BearID chips

Make sure that the bearID dataset is on your machine and then use the following
command:

```sh
make bearfacesegmentation_yolov8_generate_chips
```

To run it on other images, one can run the following command:

```sh
python ./scripts/bearfacesegmentation/chips/generate.py \
  --source-dir ./data/01_raw/BearID/images/ \
  --save-path ./data/07_model_output/bearfacesegmentation/chips/yolov8/ \
  --instance-segmentation-model-weights ./data/06_models/bearfacesegmentation/yolov8/roboflow_relabelled_current_best/weights/best.pt \
  --loglevel "info"
```

## Fast Track for bearfacesegmentation

To download the data, prepare it, train an object detector and run
inference, one can run the following command:

```sh
make bearfacesegmentation
```

___Note___: the bearfacedection command should be run prior to this
command.

## Download SAM checkpoint weights

Run the following commands:

```sh
make download_sam_weights
make download_sam_hq_weights
```

## Prepare the data to train the bearfacesegmentation models

```sh
make bearfacesegmentation_data 
```

## Train the face segmentors

```sh
make bearfacesegmentation_train
```

