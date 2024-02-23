#!/usr/bin/env bash

set -x

BROOKFALLS_GENERATED_CHIPS_ROOT_DIR="data/07_model_output/bearfacesegmentation/chips/yolov8"

BRITISH_COLUMBIA_GENERATED_CHIPS_ROOT_DIR="data/07_model_output/bearfacesegmentation/chips/generated_britishColumbia"

OUTPUT_DIR="data/07_model_output/bearfacesegmentation/chips/all/"

mkdir -p "$OUTPUT_DIR"

cp -R "$BROOKFALLS_GENERATED_CHIPS_ROOT_DIR/resized" "$OUTPUT_DIR"
cp -R "$BROOKFALLS_GENERATED_CHIPS_ROOT_DIR/raw/" "$OUTPUT_DIR"
cp -R "$BROOKFALLS_GENERATED_CHIPS_ROOT_DIR/padded/" "$OUTPUT_DIR"

cp -R "$BRITISH_COLUMBIA_GENERATED_CHIPS_ROOT_DIR/resized/" "$OUTPUT_DIR"
cp -R "$BRITISH_COLUMBIA_GENERATED_CHIPS_ROOT_DIR/raw/" "$OUTPUT_DIR"
cp -R "$BRITISH_COLUMBIA_GENERATED_CHIPS_ROOT_DIR/padded/" "$OUTPUT_DIR"
