import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "results" / "summary.json").read_text())


def test_small_model_latency_is_host_bound_not_gpu_bound():
    # a nano model at batch 1 does not follow GPU arithmetic: the 4090 host is several times slower
    assert S["yolo_n_latency_4090_ms"] > 2 * S["yolo_n_latency_5090_ms"]


def test_batching_multiplies_nano_throughput():
    assert S["batching_gain_nano_5090"] > 10


def test_sam2_outlier_is_far_from_its_prediction():
    assert S["sam2_pro6000_outlier_ms"] > 10 * S["sam2_pro6000_predicted_ms"]


def test_ranking_changes_with_the_model():
    assert S["gdino_4090_ms"] > S["gdino_3090_ms"]  # newer card, slower on this model in this data


def test_detectron2_is_slower_than_its_standin():
    assert S["detectron2_over_standin"] > 1.3
