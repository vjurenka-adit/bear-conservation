#!/usr/bin/env bash

set -x

GROUNDING_DINO_WEIGHTS_PATH="./data/06_models/GroundingDINO/"

mkdir -p "$GROUNDING_DINO_WEIGHTS_PATH"

cd "$GROUNDING_DINO_WEIGHTS_PATH" || exit

wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
