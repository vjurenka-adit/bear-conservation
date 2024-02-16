#!/usr/bin/env bash

while getopts d:t: flag; do
	case "${flag}" in
	d) datasetid=${OPTARG} ;;
	t) to=${OPTARG} ;;
	esac
done

help() {
	# Display Help
	echo "Download the Roboflow dataset."
	echo
	echo "PRIVATE_KEY should be set as an ENV."
	echo "Syntax: download_roboflow_bearfacedetection.sh  [-d|t]"
	echo "options:"
	echo "d     [mandatory] roboflow datasetid"
	echo "t     [mandatory] destination"
	echo
}

if [ -z "$datasetid" ]; then
	echo "-d not supplied."
	echo
	help
	exit 1
fi
if [ -z "$to" ]; then
	echo "-t not supplied"
	echo
	help
	exit 1
fi
if [ -z "$PRIVATE_KEY" ]; then
	echo "Please provide the env variable PRIVATE_KEY"
	echo
	help
	exit 1
fi

DESTINATION="$to/$datasetid/"

mkdir -p $DESTINATION
cd "$DESTINATION" || exit

dataset_url="https://app.roboflow.com/ds/$datasetid?key=$PRIVATE_KEY"
echo "Downloading archive file from roboflow..."
curl --progress-bar -L "$dataset_url" >roboflow.zip
unzip roboflow.zip
rm roboflow.zip
