#!/usr/bin/env bash

ROBOFLOW_DATASET_ID="b8vuUrGhDn"
DESTINATION="data/05_model_input/bearfacedetection/relabelled/$ROBOFLOW_DATASET_ID/"

mkdir -p $DESTINATION
cd "$DESTINATION" || exit

if [ -n "$PRIVATE_KEY" ]; then
	dataset_url="https://app.roboflow.com/ds/$ROBOFLOW_DATASET_ID?key=$PRIVATE_KEY"
	echo "Downloading archive file from roboflow..."
	curl --progress-bar -L "$dataset_url" >roboflow.zip
	unzip roboflow.zip
	rm roboflow.zip
else
	echo "Please provide the env variable PRIVATE_KEY"
fi
