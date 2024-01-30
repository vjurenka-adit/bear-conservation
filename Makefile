.PHONY: download_dataset install_dependencies install_local_packages setup dev_notebook

install_dependencies:
	python -m pip install -r requirements.txt

install_local_packages:
	python -m pip install -e .

setup: install_dependencies install_local_packages

dev_notebook:
	jupyter lab

download_dataset:
	./scripts/data/download_dataset.sh
