.PHONY: install download-models prepare-train prepare-val train infer fine-pairs evaluate test

PYTHON ?= python
CONFIG ?= configs/paper.yaml
NUSCENES_ROOT ?= data/nuscenes
MODELS_URL ?= https://drive.google.com/drive/folders/1p133pqV2i6RiZHF30qR6LCDdoSuAWILv
CHECKPOINT ?= models/coarse_4frame_caf.pth
NEIGHBOR_FRAMES ?= 3

install:
	$(PYTHON) -m pip install -e '.[dev]'

download-models:
	$(PYTHON) -m pip install gdown
	$(PYTHON) -m gdown --folder $(MODELS_URL) -O models

prepare-train:
	$(PYTHON) scripts/prepare_nuscenes.py --dataroot $(NUSCENES_ROOT) --split train

prepare-val:
	$(PYTHON) scripts/prepare_nuscenes.py --dataroot $(NUSCENES_ROOT) --split val

train:
	$(PYTHON) scripts/train_coarse.py --config $(CONFIG) --output checkpoints/coarse.pt

infer:
	$(PYTHON) scripts/infer_coarse.py --config $(CONFIG) --checkpoint $(CHECKPOINT) --neighbor-frames $(NEIGHBOR_FRAMES) --output outputs/coarse_predictions.json

fine-pairs:
	$(PYTHON) scripts/prepare_fine_pairs.py outputs/coarse_predictions.json --output outputs/fine_pairs

evaluate:
	$(PYTHON) scripts/evaluate.py outputs/distance_errors.json

test:
	$(PYTHON) -m pytest -q
	$(PYTHON) -m compileall -q src scripts
