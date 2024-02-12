.PHONY: download_dataset install_dependencies install_local_packages setup flattened_bearid_images
	dev_notebook bearface_data_yolov8_txt_format data_bearfacedetection
	download_sam_weights

install_dependencies: requirements.txt
	python -m pip install -r requirements.txt

install_local_packages:
	python -m pip install -e .

setup: install_dependencies install_local_packages

dev_notebook:
	jupyter lab

download_dataset:
	./scripts/data/download_dataset.sh

# bearfacedetection

download_roboflow_bearfacedetection:
	./scripts/data/download_roboflow_bearfacedetection.sh

data_bearid_build_metadata:
	python ./scripts/data/build_metadata_bearid.py \
		--bearid-base-path ./data/01_raw/BearID/ \
		--to ./data/03_primary/golden_dataset/ \
		--loglevel "info"

data: download_dataset data_bearid_build_metadata

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

bearfacedetection_data: bearfacedetection_data_golden_dataset_yolov8_txt_format

bearfacedetection_train_baseline_golden_dataset:
	python ./scripts/bearfacedetection/train.py \
		--data ./data/05_model_input/bearfacedetection/golden_dataset/data.yaml \
		--epochs 2 \
		--experiment-name golden_dataset_baseline \
		--model "yolov8n.pt" \
		--loglevel "info"

bearfacedetection_train_baseline_roboflow:
	python ./scripts/bearfacedetection/train.py \
		--data ./data/05_model_input/bearfacedetection/relabelled/b8vuUrGhDn/data.yaml \
		--epochs 2 \
		--experiment-name roboflow_b8vuUrGhDn_baseline \
		--model "yolov8n.pt" \
		--loglevel "info"

bearfacedetection_predict_baseline_golden_dataset:
	python ./scripts/bearfacedetection/predict.py \
	  --model-weights data/06_models/bearfacedetection/yolov8/golden_dataset_baseline/weights/best.pt \
	  --source-path data/05_model_input/bearfacedetection/golden_dataset/val/images/ \
	  --save-path data/07_model_output/bearfacedetection/golden_dataset/val/predictions/

bearfacedetection: bearfacedetection_data bearfacedetection_train_baseline_golden_dataset bearfacedetection_predict_baseline_golden_dataset

# bearfacesegmentation

download_sam_weights:
	./scripts/bearfacesegmentation/sam/download_checkpoint.sh

download_sam_hq_weights:
	./scripts/bearfacesegmentation/sam-hq/download_checkpoint.sh

segment_sam_golden_dataset_bear_bodies:
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

segment_sam_hq_golden_dataset_bear_bodies:
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

segment_sam_golden_dataset_bear_heads:
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

segment_sam_with_relabelled_roboflow_bear_heads:
	python ./scripts/bearfacesegmentation/sam/segment_head.py \
	  --from-body-masks ./data/04_feature/bearfacesegmentation/sam/body/train/ \
	  --from-head-bbox-yolov8-labels ./data/05_model_input/bearfacedetection/relabelled/b8vuUrGhDn/train/ \
	  --to ./data/04_feature/bearfacesegmentation/b8vuUrGhDn/sam/head/train/ \
	  --loglevel "info"
	python ./scripts/bearfacesegmentation/sam/segment_head.py \
	  --from-body-masks ./data/04_feature/bearfacesegmentation/sam/body/test/ \
	  --from-head-bbox-yolov8-labels ./data/05_model_input/bearfacedetection/relabelled/b8vuUrGhDn/valid/ \
	  --to ./data/04_feature/bearfacesegmentation/b8vuUrGhDn/sam/head/test/ \
	  --loglevel "info"

segment_sam_hq_golden_dataset_bear_heads:
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
	  --from-head-masks ./data/04_feature/bearfacesegmentation/b8vuUrGhDn/sam/head/test/ \
	  --to ./data/04_feature/bearfacesegmentation/yolov8_txt_format/b8vuUrGhDn/test \
	  --loglevel "info"
	python ./scripts/bearfacesegmentation/data/build_yolov8_txt_format.py \
	  --from-head-masks ./data/04_feature/bearfacesegmentation/b8vuUrGhDn/sam/head/train/ \
	  --to ./data/04_feature/bearfacesegmentation/yolov8_txt_format/b8vuUrGhDn/train \
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
	  --yolov8-txt-format ./data/04_feature/bearfacesegmentation/yolov8_txt_format/b8vuUrGhDn/ \
	  --from-roboflow \
	  --to ./data/05_model_input/bearfacesegmentation/b8vuUrGhDn/ \
	  --loglevel "info"

bearfacesegmentation_train_baseline_golden_dataset:
	python ./scripts/bearfacesegmentation/train.py \
		--data ./data/05_model_input/bearfacesegmentation/v0/data.yaml \
		--epochs 2 \
		--experiment-name golden_dataset_baseline \
		--model "yolov8n-seg.pt" \
		--loglevel "info"

bearfacesegmentation_train_baseline_roboflow_relabelled:
	python ./scripts/bearfacesegmentation/train.py \
		--data ./data/05_model_input/bearfacesegmentation/b8vuUrGhDn/data.yaml \
		--epochs 2 \
		--experiment-name roboflow_relabelled_baseline \
		--model "yolov8n-seg.pt" \
		--loglevel "info"

# bearfacelandmarkdetection

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

bearfacelandmarkdetection_train_baseline_golden_dataset:
	python ./scripts/bearfacelandmarkdetection/train.py \
		--data ./data/05_model_input/bearfacelandmarkdetection/golden_dataset/data.yaml \
		--epochs 2 \
		--experiment-name golden_dataset_baseline \
		--model "yolov8n-pose.pt" \
		--loglevel "info"
