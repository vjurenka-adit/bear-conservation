#!/usr/bin/env bash

set -x

INPUT_CHIPS_PATH="./data/09_external/artifacts"
ZIP_ARCHIVE="${INPUT_CHIPS_PATH}/chips.zip"
OUTPUT_CHIPS_PATH="./data/07_model_output/bearfacesegmentation"

if [ ! -f "$ZIP_ARCHIVE" ]; then
	echo "$ZIP_ARCHIVE does not exist."
	exit 1
fi

unzip "$ZIP_ARCHIVE" -d "$OUTPUT_CHIPS_PATH"
