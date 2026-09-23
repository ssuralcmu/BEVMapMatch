.PHONY: install prepare-train prepare-val train infer fine-pairs evaluate test

PYTHON ?= python
CONFIG ?= configs/paper.yaml
NUSCENES_ROOT ?= data/nuscenes

install:
	$(PYTHON) -m pip install -e '.[dev]'

prepare-train:
	$(PYTHON) scripts/prepare_nuscenes.py --dataroot $(NUSCENES_ROOT) --split train

prepare-val:
	$(PYTHON) scripts/prepare_nuscenes.py --dataroot $(NUSCENES_ROOT) --split val

train:
	$(PYTHON) scripts/train_coarse.py --config $(CONFIG) --output checkpoints/coarse.pt

infer:
	$(PYTHON) scripts/infer_coarse.py --config $(CONFIG) --checkpoint checkpoints/coarse.pt

fine-pairs:
	$(PYTHON) scripts/prepare_fine_pairs.py outputs/coarse_predictions.json --output outputs/fine_pairs

evaluate:
	$(PYTHON) scripts/evaluate.py outputs/distance_errors.json

test:
	$(PYTHON) -m pytest -q
	$(PYTHON) -m compileall -q src scripts

