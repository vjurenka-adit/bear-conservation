#!/usr/bin/env bash

set -x

rclone --drive-shared-with-me --progress copy remote:"AI for Bears Shared/01. Data/BearID" "data/01_raw/BearID"

rclone --drive-shared-with-me --progress copy remote:"AI for Bears Shared/01. Data/Hack the Planet" "data/01_raw/Hack the Planet"

rclone --drive-shared-with-me --progress copy remote:"AI for Bears Shared/01. Data/SLU Multi-Species" "data/01_raw/SLU Multi-Species"

rclone --drive-shared-with-me --progress copy remote:"AI for Bears Shared/01. Data/Sensing Clues" "Sensing Clues"
