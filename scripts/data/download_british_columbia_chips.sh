#!/usr/bin/env bash

set -x

rclone --drive-shared-with-me --progress copy remote:"AI for Bears Shared/05. Subgroup folder/face_detection_and_segmentation/generated_britishColumbia/" "data/07_model_output/bearfacesegmentation/chips/generated_britishColumbia/"
