import numpy as np
import torch

from bevmapmatch.data import perturbation_to_pixel
from bevmapmatch.fine import map_query_center
from bevmapmatch.losses import gaussian_targets
from bevmapmatch.metrics import coarse_metrics, localization_metrics


def test_coordinate_convention():
    assert perturbation_to_pixel([10, -20], 500, 500, 2) == (540.0, 480.0)


def test_coarse_metrics():
    logits = torch.zeros(2, 100); labels = torch.zeros(2, 100)
    logits[0, 22] = 1; labels[0, 22] = 1
    logits[1, 44] = 1; labels[1, 55] = 1
    assert coarse_metrics(logits, labels) == {"top_1x1": 0.5, "top_3x3": 1.0}


def test_soft_targets_are_normalized():
    labels = torch.zeros(1, 100); labels[0, 55] = 1
    assert torch.allclose(gaussian_targets(labels, 10, 0.8).sum(1), torch.ones(1))


def test_identity_homography():
    assert map_query_center(np.eye(3), (101, 201), (101, 201)) == (100.0, 50.0)


def test_recall():
    metrics = localization_metrics([0.5, 1.5, 3.0])
    assert metrics["recall_at_1m"] == 1 / 3
    assert metrics["recall_at_2m"] == 2 / 3

