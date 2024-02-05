#!/usr/bin/env bash

set -x

WEIGHTS_PATH="./data/06_models/bearfacesegmentation/sam/weights/"
WEIGHTS_FILE="sam_vit_h_4b8939.pth"

mkdir -p "$WEIGHTS_PATH"
wget "https://dl.fbaipublicfiles.com/segment_anything/$WEIGHTS_FILE" -P "$WEIGHTS_PATH"
