#!/usr/bin/env bash

set -x

LIGHTGLUE_VENDOR_PATH="vendors/LightGlue"

mkdir -p "$LIGHTGLUE_VENDOR_PATH"
git clone https://github.com/cvg/LightGlue/ "$LIGHTGLUE_VENDOR_PATH"
cd "$LIGHTGLUE_VENDOR_PATH" || exit

python -m pip install -e .
