#!/usr/bin/env bash

set -x

WEIGHTS_PATH="./data/06_models/bearfacesegmentation/sam-hq/weights/"
WEIGHTS_FILE="sam_hq_vit_h.pth"

mkdir -p "$WEIGHTS_PATH"
wget "https://huggingface.co/lkeab/hq-sam/resolve/main/$WEIGHTS_FILE" -P "$WEIGHTS_PATH"
