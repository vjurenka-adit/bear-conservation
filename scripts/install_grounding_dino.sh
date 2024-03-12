#!/usr/bin/env bash

set -x

GROUNDING_DINO_VENDOR_PATH="vendors/GroundingDINO"

mkdir -p "$GROUNDING_DINO_VENDOR_PATH"
git clone https://github.com/IDEA-Research/GroundingDINO.git "$GROUNDING_DINO_VENDOR_PATH"
cd "$GROUNDING_DINO_VENDOR_PATH" || exit

python -m pip install -e .

mkdir -p weights
cd weights || exit

wget --show-progress -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
