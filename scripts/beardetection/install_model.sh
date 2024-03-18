#!/usr/bin/env bash

set -x

ARTIFACT_FILEPATH=./data/09_external/artifacts/beardetection/weights/model.pt
MODEL_DIR=./data/06_models/beardetection/model/weights/

ls -larth $ARTIFACT_FILEPATH

mkdir -p "$MODEL_DIR"

cp $ARTIFACT_FILEPATH $MODEL_DIR
