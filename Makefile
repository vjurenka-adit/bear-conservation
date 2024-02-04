.PHONY: download_dataset install_dependencies install_local_packages setup dev_notebook bearface_data_yolov8_txt_format data_bearfacedetection

install_dependencies: requirements.txt
	python -m pip install -r requirements.txt

install_local_packages:
	python -m pip install -e .

setup: install_dependencies install_local_packages

dev_notebook:
	jupyter lab

bearfacedetection_data_golden_dataset_yolov8_txt_format:
	python ./scripts/bearfacedetection/data/build_yolov8_txt_format.py \
		--xml-filepath ./data/01_raw/BearID/images_train_without_bc.xml \
		--to ./data/04_feature/bearfacedetection/golden_dataset/train/ \
		--loglevel "info"
	python ./scripts/bearfacedetection/data/build_yolov8_txt_format.py \
		--xml-filepath ./data/01_raw/BearID/images_test_without_bc.xml \
		--to ./data/04_feature/bearfacedetection/golden_dataset/test/ \
		--loglevel "info"

data_bearfacedetection: bearfacedetection_data_golden_dataset_yolov8_txt_format

download_dataset:
	./scripts/data/download_dataset.sh
