.PHONY: download_dataset install_dependencies install_local_packages setup flattened_bearid_images
	dev_notebook bearface_data_yolov8_txt_format data_bearfacedetection
	download_sam_weights bearfacedetection bearfacesegmentation bearfacelandmarkdetection
	install_vendor_packages

ifeq ($(OS),Windows_NT)
    platform := Windows
else
    platform := $(shell uname -s)
endif

ifeq ($(platform), Darwin)
    platform_specific_requirements = requirements/macos.txt
else
	platform_specific_requirements = requirements/linux.txt
endif

install_dependencies: $(platform_specific_requirements)
	python -m pip install -r $(platform_specific_requirements)

install_local_packages:
	python -m pip install -e .

install_vendor_packages:
	./scripts/install_lightglue.sh

setup: install_dependencies install_local_packages install_vendor_packages

dev_notebook:
	jupyter lab

download_dataset:
	./scripts/data/download_dataset.sh

download_roboflow_bearfacedetection:
	./scripts/data/download_roboflow_bearfacedetection.sh \
		-t "./data/05_model_input/bearfacedetection/relabelled" \
		-d "MS25RkYkMA"


data_bearid_build_metadata:
	python ./scripts/data/build_metadata_bearid.py \
		--bearid-base-path ./data/01_raw/BearID/ \
		--to ./data/03_primary/golden_dataset/ \
		--loglevel "info"

data: download_dataset download_roboflow_bearfacedetection data_bearid_build_metadata

# -----------------
# bearfacedetection
# -----------------

bearfacedetection_data_golden_dataset_yolov8_txt_format:
	python ./scripts/bearfacedetection/data/build_yolov8_txt_format.py \
		--xml-filepath ./data/01_raw/BearID/images_train_without_bc.xml \
		--to ./data/04_feature/bearfacedetection/golden_dataset/train/ \
		--loglevel "info"
	python ./scripts/bearfacedetection/data/build_yolov8_txt_format.py \
		--xml-filepath ./data/01_raw/BearID/images_test_without_bc.xml \
		--to ./data/04_feature/bearfacedetection/golden_dataset/test/ \
		--loglevel "info"

bearfacedetection_data_golden_dataset_build_model_input:
	python ./scripts/bearfacedetection/data/build_model_input.py \
		--from ./data/04_feature/bearfacedetection/golden_dataset/ \
		--to ./data/05_model_input/bearfacedetection/golden_dataset/ \
		--loglevel "info"

bearfacedetection_data: bearfacedetection_data_golden_dataset_yolov8_txt_format bearfacedetection_data_golden_dataset_build_model_input

bearfacedetection_train_baseline_golden_dataset:
	python ./scripts/bearfacedetection/train.py \
		--data ./data/05_model_input/bearfacedetection/golden_dataset/data.yaml \
		--epochs 2 \
		--experiment-name golden_dataset_baseline \
		--model "yolov8n.pt" \
		--loglevel "info"

bearfacedetection_train_baseline_roboflow:
	python ./scripts/bearfacedetection/train.py \
		--data ./data/05_model_input/bearfacedetection/relabelled/MS25RkYkMA/data.yaml \
		--epochs 2 \
		--experiment-name roboflow_MS25RkYkMA_baseline \
		--model "yolov8n.pt" \
		--loglevel "info"

bearfacedetection_train: bearfacedetection_train_baseline_golden_dataset bearfacedetection_train_baseline_roboflow

bearfacedetection_predict_baseline_golden_dataset:
	python ./scripts/bearfacedetection/predict.py \
	  --model-weights data/06_models/bearfacedetection/yolov8/golden_dataset_baseline/weights/best.pt \
	  --source-path data/05_model_input/bearfacedetection/golden_dataset/val/images/ \
	  --save-path data/07_model_output/bearfacedetection/golden_dataset/val/predictions/

bearfacedetection_predict: bearfacedetection_predict_baseline_golden_dataset

# Command that runs all bearfacedetection code to prepare the data,
# train and run inference
bearfacedetection: bearfacedetection_data bearfacedetection_train bearfacedetection_predict

# --------------------
# bearfacesegmentation
# --------------------

download_sam_weights:
	./scripts/bearfacesegmentation/sam/download_checkpoint.sh

download_sam_hq_weights:
	./scripts/bearfacesegmentation/sam-hq/download_checkpoint.sh

bearfacesegmentation_segment_sam_golden_dataset_bear_bodies:
	python ./scripts/bearfacesegmentation/sam/segment_body.py \
	  --model-weights ./data/06_models/bearfacesegmentation/sam/weights/sam_vit_h_4b8939.pth \
	  --to ./data/04_feature/bearfacesegmentation/sam/body/train/ \
	  --xml-filepath ./data/01_raw/BearID/images_train_without_bc.xml \
    	  --loglevel "info"
	python ./scripts/bearfacesegmentation/sam/segment_body.py \
	  --model-weights ./data/06_models/bearfacesegmentation/sam/weights/sam_vit_h_4b8939.pth \
	  --to ./data/04_feature/bearfacesegmentation/sam/body/test/ \
	  --xml-filepath ./data/01_raw/BearID/images_test_without_bc.xml \
    	  --loglevel "info"

bearfacesegmentation_segment_sam_hq_golden_dataset_bear_bodies:
	python ./scripts/bearfacesegmentation/sam-hq/segment_body.py \
	  --model-weights ./data/06_models/bearfacesegmentation/sam-hq/weights/sam_hq_vit_h.pth \
	  --to ./data/04_feature/bearfacesegmentation/sam-hq/body/train/ \
	  --xml-filepath ./data/01_raw/BearID/images_train_without_bc.xml \
	  --loglevel "info"
	python ./scripts/bearfacesegmentation/sam-hq/segment_body.py \
	  --model-weights ./data/06_models/bearfacesegmentation/sam-hq/weights/sam_hq_vit_h.pth \
	  --to ./data/04_feature/bearfacesegmentation/sam-hq/body/test/ \
	  --xml-filepath ./data/01_raw/BearID/images_test_without_bc.xml \
	  --loglevel "info"

bearfacesegmentation_segment_sam_golden_dataset_bear_heads:
	python ./scripts/bearfacesegmentation/sam/segment_head.py \
	  --from-body-masks ./data/04_feature/bearfacesegmentation/sam/body/train/ \
	  --from-head-bbox-xml-filepath ./data/01_raw/BearID/images_train_without_bc.xml \
	  --to ./data/04_feature/bearfacesegmentation/sam/head/train/ \
	--loglevel "info"
	python ./scripts/bearfacesegmentation/sam/segment_head.py \
	  --from-body-masks ./data/04_feature/bearfacesegmentation/sam/body/test/ \
	  --from-head-bbox-xml-filepath ./data/01_raw/BearID/images_test_without_bc.xml \
	  --to ./data/04_feature/bearfacesegmentation/sam/head/test/ \
	--loglevel "info"

bearfacesegmentation_segment_sam_hq_golden_dataset_bear_heads:
	python ./scripts/bearfacesegmentation/sam-hq/segment_head.py \
	  --from-body-masks ./data/04_feature/bearfacesegmentation/sam-hq/body/train/ \
	  --from-head-bbox-xml-filepath ./data/01_raw/BearID/images_train_without_bc.xml \
	  --to ./data/04_feature/bearfacesegmentation/sam-hq/head/train/ \
	  --loglevel "info"
	python ./scripts/bearfacesegmentation/sam-hq/segment_head.py \
	  --from-body-masks ./data/04_feature/bearfacesegmentation/sam-hq/body/test/ \
	  --from-head-bbox-xml-filepath ./data/01_raw/BearID/images_test_without_bc.xml \
	  --to ./data/04_feature/bearfacesegmentation/sam-hq/head/test/ \
	--loglevel "info"

bearfacesegmentation_segment_sam_with_relabelled_roboflow_bear_heads:
	python ./scripts/bearfacesegmentation/sam/segment_head.py \
	  --from-body-masks ./data/04_feature/bearfacesegmentation/sam/body/train/ \
	  --from-head-bbox-yolov8-labels ./data/05_model_input/bearfacedetection/relabelled/MS25RkYkMA/train/ \
	  --to ./data/04_feature/bearfacesegmentation/MS25RkYkMA/sam/head/train/ \
	  --loglevel "info"
	python ./scripts/bearfacesegmentation/sam/segment_head.py \
	  --from-body-masks ./data/04_feature/bearfacesegmentation/sam/body/test/ \
	  --from-head-bbox-yolov8-labels ./data/05_model_input/bearfacedetection/relabelled/MS25RkYkMA/valid/ \
	  --to ./data/04_feature/bearfacesegmentation/MS25RkYkMA/sam/head/test/ \
	  --loglevel "info"

bearfacesegmentation_data_golden_dataset_yolov8_txt_format:
	python ./scripts/bearfacesegmentation/data/build_yolov8_txt_format.py \
	  --from-head-masks ./data/04_feature/bearfacesegmentation/sam/head/test/ \
	  --to ./data/04_feature/bearfacesegmentation/yolov8_txt_format/v0/test \
	  --loglevel "info"
	python ./scripts/bearfacesegmentation/data/build_yolov8_txt_format.py \
	  --from-head-masks ./data/04_feature/bearfacesegmentation/sam/head/train/ \
	  --to ./data/04_feature/bearfacesegmentation/yolov8_txt_format/v0/train \
	  --loglevel "info"

bearfacesegmentation_data_roboflow_relabelled_yolov8_txt_format:
	python ./scripts/bearfacesegmentation/data/build_yolov8_txt_format.py \
	  --from-head-masks ./data/04_feature/bearfacesegmentation/MS25RkYkMA/sam/head/test/ \
	  --to ./data/04_feature/bearfacesegmentation/yolov8_txt_format/MS25RkYkMA/test \
	  --loglevel "info"
	python ./scripts/bearfacesegmentation/data/build_yolov8_txt_format.py \
	  --from-head-masks ./data/04_feature/bearfacesegmentation/MS25RkYkMA/sam/head/train/ \
	  --to ./data/04_feature/bearfacesegmentation/yolov8_txt_format/MS25RkYkMA/train \
	  --loglevel "info"

bearfacesegmentation_data_golden_dataset_build_model_input:
	python ./scripts/bearfacesegmentation/data/build_model_input.py \
	  --split-metadata-yaml ./data/03_primary/golden_dataset/metadata.yaml \
	  --yolov8-txt-format ./data/04_feature/bearfacesegmentation/yolov8_txt_format/v0/ \
	  --to ./data/05_model_input/bearfacesegmentation/v0/ \
	  --loglevel "info"

bearfacesegmentation_data_roboflow_relabelled_build_model_input:
	python ./scripts/bearfacesegmentation/data/build_model_input.py \
	  --split-metadata-yaml ./data/03_primary/golden_dataset/metadata.yaml \
	  --yolov8-txt-format ./data/04_feature/bearfacesegmentation/yolov8_txt_format/MS25RkYkMA/ \
	  --from-roboflow \
	  --to ./data/05_model_input/bearfacesegmentation/MS25RkYkMA/ \
	  --loglevel "info"

bearfacesegmentation_data: download_sam_weights download_sam_hq_weights bearfacesegmentation_segment_sam_golden_dataset_bear_bodies bearfacesegmentation_segment_sam_hq_golden_dataset_bear_bodies bearfacesegmentation_segment_sam_golden_dataset_bear_heads bearfacesegmentation_segment_sam_hq_golden_dataset_bear_heads bearfacesegmentation_segment_sam_with_relabelled_roboflow_bear_heads bearfacesegmentation_data_golden_dataset_yolov8_txt_format bearfacesegmentation_data_roboflow_relabelled_yolov8_txt_format bearfacesegmentation_data_golden_dataset_build_model_input bearfacesegmentation_data_roboflow_relabelled_build_model_input

bearfacesegmentation_train_baseline_golden_dataset:
	python ./scripts/bearfacesegmentation/train.py \
		--data ./data/05_model_input/bearfacesegmentation/v0/data.yaml \
		--epochs 2 \
		--experiment-name golden_dataset_baseline \
		--model "yolov8n-seg.pt" \
		--loglevel "info"

bearfacesegmentation_train_baseline_roboflow_relabelled:
	python ./scripts/bearfacesegmentation/train.py \
		--data ./data/05_model_input/bearfacesegmentation/MS25RkYkMA/data.yaml \
		--epochs 2 \
		--experiment-name roboflow_relabelled_baseline \
		--model "yolov8n-seg.pt" \
		--loglevel "info"

bearfacesegmentation_train_current_best_roboflow_relabelled:
	python ./scripts/bearfacesegmentation/train.py \
		--data ./data/05_model_input/bearfacesegmentation/MS25RkYkMA/data.yaml \
		--epochs 40 \
		--close-mosaic 10 \
		--batch 64 \
		--imgsz 1024 \
		--degrees 25 \
		--experiment-name roboflow_relabelled_current_best \
		--model "yolov8n-seg.pt" \
		--loglevel "info"

bearfacesegmentation_train: bearfacesegmentation_train_baseline_golden_dataset bearfacesegmentation_train_baseline_roboflow_relabelled

bearfacesegmentation_predict_baseline_golden_dataset:
	python ./scripts/bearfacesegmentation/predict.py \
	  --model-weights data/06_models/bearfacesegmentation/yolov8/golden_dataset_baseline/weights/best.pt \
	  --source-path data/05_model_input/bearfacesegmentation/v0/val/images/ \
	  --save-path data/07_model_output/bearfacesegmentation/golden_dataset/val/predictions/

bearfacesegmentation_predict_baseline_roboflow_relabelled:
	python ./scripts/bearfacesegmentation/predict.py \
	  --model-weights ./data/06_models/bearfacesegmentation/yolov8/roboflow_relabelled_baseline/weights/best.pt \
	  --source-path ./data/05_model_input/bearfacesegmentation/MS25RkYkMA/val/images/ \
	  --save-path data/07_model_output/bearfacesegmentation/MS25RkYkMA/baseline/val/predictions/

bearfacesegmentation_predict: bearfacesegmentation_predict_baseline_golden_dataset bearfacesegmentation_predict_baseline_roboflow_relabelled

bearfacesegmentation_yolov8_generate_chips:
	python ./scripts/bearfacesegmentation/chips/generate.py \
	  --source-dir ./data/01_raw/BearID/images/ \
	  --save-path ./data/07_model_output/bearfacesegmentation/chips/yolov8/ \
	  --instance-segmentation-model-weights ./data/06_models/bearfacesegmentation/yolov8/roboflow_relabelled_current_best/weights/best.pt \
	  --loglevel "info"

bearfacesegmentation_yolov8_generate_test_chips:
	python ./scripts/bearfacesegmentation/chips/generate.py \
	  --source-dir ./data/09_external/images/bears/ \
	  --save-path ./data/07_model_output/bearfacesegmentation/chips/test/ \
	  --instance-segmentation-model-weights ./data/06_models/bearfacesegmentation/yolov8/roboflow_relabelled_current_best/weights/best.pt \
	  --loglevel "info"

bearfacesegmentation_archive_chips:
	python ./scripts/bearfacesegmentation/chips/archive.py \
	  --source-dir ./data/07_model_output/bearfacesegmentation/chips/yolov8/resized/ \
	  --save-path ./data/07_model_output/bearfacesegmentation/chips/yolov8/ \
	  --loglevel "info"

bearfacesegmentation: bearfacesegmentation_data bearfacesegmentation_train bearfacesegmentation_predict

# -------------------------
# bearfacelandmarkdetection
# -------------------------

bearfacelandmarkdetection_data_golden_dataset_yolov8_txt_format:
	python ./scripts/bearfacelandmarkdetection/data/build_yolov8_txt_format.py \
		--xml-filepath ./data/01_raw/BearID/images_train_without_bc.xml \
		--to ./data/04_feature/bearfacelandmarkdetection/golden_dataset/train/ \
		--loglevel "info"
	python ./scripts/bearfacelandmarkdetection/data/build_yolov8_txt_format.py \
		--xml-filepath ./data/01_raw/BearID/images_test_without_bc.xml \
		--to ./data/04_feature/bearfacelandmarkdetection/golden_dataset/test/ \
		--loglevel "info"

bearfacelandmarkdetection_data_golden_dataset_build_model_input:
	python ./scripts/bearfacelandmarkdetection/data/build_model_input.py \
		--from ./data/04_feature/bearfacelandmarkdetection/golden_dataset/ \
		--to ./data/05_model_input/bearfacelandmarkdetection/golden_dataset/ \
		--loglevel "info"

bearfacelandmarkdetection_data: bearfacelandmarkdetection_data_golden_dataset_yolov8_txt_format bearfacelandmarkdetection_data_golden_dataset_build_model_input

bearfacelandmarkdetection_train_baseline_golden_dataset:
	python ./scripts/bearfacelandmarkdetection/train.py \
		--data ./data/05_model_input/bearfacelandmarkdetection/golden_dataset/data.yaml \
		--epochs 2 \
		--experiment-name golden_dataset_baseline \
		--model "yolov8n-pose.pt" \
		--loglevel "info"

bearfacelandmarkdetection_train: bearfacelandmarkdetection_train_baseline_golden_dataset

bearfacelandmarkdetection: bearfacelandmarkdetection_data bearfacelandmarkdetection_train

bearfacelandmarkdetection_predict_baseline_golden_dataset:
	python ./scripts/bearfacelandmarkdetection/predict.py \
	  --model-weights data/06_models/bearfacelandmarkdetection/yolov8/golden_dataset_baseline/weights/best.pt \
	  --source-path data/05_model_input/bearfacelandmarkdetection/golden_dataset/val/images/ \
	  --save-path data/07_model_output/bearfacelandmarkdetection/golden_dataset/val/predictions/

bearfacelandmarkdetection_predict: bearfacelandmarkdetection_predict_baseline_golden_dataset


# ------------------
# bearidentification
# ------------------

bearidentification_data_download_generated_chips_british_columbia:
	./scripts/data/download_british_columbia_chips.sh

bearidentification_data_merge_generated_chips:
	./scripts/data/merge_generated_chips.sh


bearidentification_data_split_by_individual:
	python ./scripts/bearidentification/data/split/by_individual.py \
	  --save-path ./data/04_feature/bearidentification/bearid/split/ \
	  --chips-root-dir ./data/07_model_output/bearfacesegmentation/chips/yolov8/resized/square_dim_300/ \
	  --loglevel "info"

bearidentification_data_lightglue_keypoints_generate:
	python ./scripts/bearidentification/lightglue/keypoints/generation.py \
	  --extractor "sift" \
	  --loglevel "info"
	python ./scripts/bearidentification/lightglue/keypoints/generation.py \
	  --extractor "superpoint" \
	  --loglevel "info"
	python ./scripts/bearidentification/lightglue/keypoints/generation.py \
	  --extractor "disk" \
	  --loglevel "info"
	python ./scripts/bearidentification/lightglue/keypoints/generation.py \
	  --extractor "aliked" \
	  --loglevel "info"
