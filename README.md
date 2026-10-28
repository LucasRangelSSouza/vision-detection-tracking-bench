# Vision: detection, tracking and segmentation bench

YOLO11, ByteTrack, SAM 2, Grounding DINO and Mask R-CNN run on real GPUs with per-image latency, throughput and VRAM recorded.

Author: Lucas Rangel. All numbers below come from `evidence/results.json` (measured runs on rented Vast.ai GPUs, Oct 2026). Rows with failed or install-only status are not included. Every row is a real run on a shared marketplace host: values are single-session medians, outliers are kept (not filtered), and a different host of the same GPU model can differ noticeably.

## Conclusion

Median latency per image (p50): YOLO11n at 640 px, batch 1 - GB10: 3.26 ms, RTX 5090: 3.59 ms, L40S: 4.48 ms, H100 NVL: 6.35 ms, RTX PRO 6000 Blackwell Server Edition: 9.22 ms, A100-PCIE-40GB: 9.56 ms, RTX 3090: 10.14 ms, RTX 4090: 16.77 ms. SAM 2 hiera_large - RTX 5090: 21.75 ms, A100-PCIE-40GB: 47.85 ms, RTX 4090: 54.11 ms, L40S: 63.37 ms, RTX 3090: 69.53 ms, GB10: 92.03 ms, RTX PRO 6000 Blackwell Server Edition: 505.52 ms. Grounding DINO base - RTX 5090: 45.36 ms, RTX PRO 6000 Blackwell Server Edition: 62.21 ms, L40S: 74.01 ms, A100-PCIE-40GB: 95.68 ms, RTX 3090: 108.05 ms, RTX 4090: 149.26 ms, GB10: 168.8 ms. Mask R-CNN R50-FPN (torchvision proxy for Detectron2) - RTX PRO 6000 Blackwell Server Edition: 13.52 ms, RTX 5090: 17.73 ms, L40S: 18.36 ms, RTX 4090: 18.6 ms, RTX 3090: 30.2 ms, GB10: 76.84 ms. Tracking (ByteTrack) runs on top of YOLO11n at about 60 fps on a single GPU. No tracking ground truth was available, so HOTA/IDF1 were not evaluated; only throughput and ID counts are reported. Detectron2 itself built on an RTX 3090 in a later sandbox suite and ran on real COCO images: Mask R-CNN 51.1 ms at the median (95th percentile 91.3 ms) and Faster R-CNN 44.5 ms, against 30.2 ms for the torchvision stand-in on random noise on the same card, so the stand-in rows are a floor. The 505.5 ms SAM 2 value on one RTX PRO 6000 host is an unexplained outlier; the same model of card is expected near 30 ms (see the Kaggle article on SAM 2, Grounding DINO and Mask R-CNN latency).

## Measured rows (status ok only)

### grounding_dino (7 rows)

| model | gpu | p50_ms | p95_ms | vram_gb |
|---|---|---|---|---|
| IDEA-Research/grounding-dino-base | RTX 4090 | 149 | 171 | 1.06 |
| IDEA-Research/grounding-dino-base | RTX 5090 | 45.4 | 92.7 | 1.08 |
| IDEA-Research/grounding-dino-base | L40S | 74 | 75.5 | 1.06 |
| IDEA-Research/grounding-dino-base | A100-PCIE-40GB | 95.7 | 189 | 1.06 |
| IDEA-Research/grounding-dino-base | RTX 3090 | 108 | 109 | 1.06 |
| IDEA-Research/grounding-dino-base | RTX PRO 6000 Blackwell Server Edition | 62.2 | 64.7 | 1.08 |
| IDEA-Research/grounding-dino-base | GB10 | 169 | 170 | 1.08 |

### maskrcnn_torchvision (7 rows)

| model | gpu | p50_ms | p95_ms | vram_gb |
|---|---|---|---|---|
| maskrcnn_resnet50_fpn | L40S | 18.4 | 18.6 | 0.62 |
| maskrcnn_resnet50_fpn | RTX 3090 | 29.9 | 30.3 | 0.62 |
| maskrcnn_resnet50_fpn | RTX 3090 | 30.5 | 32.6 | 0.62 |
| maskrcnn_resnet50_fpn | RTX 4090 | 18.6 | 19.1 | 0.62 |
| maskrcnn_resnet50_fpn | RTX 5090 | 17.7 | 17.9 | 0.65 |
| maskrcnn_resnet50_fpn | RTX PRO 6000 Blackwell Server Edition | 13.5 | 14.7 | 0.65 |
| maskrcnn_resnet50_fpn | GB10 | 76.8 | 77.8 | 1.45 |

### sam2 (7 rows)

| model | gpu | p50_ms | p95_ms | vram_gb |
|---|---|---|---|---|
| hiera_large | RTX 4090 | 54.1 | 54.8 | 1.67 |
| hiera_large | RTX 5090 | 21.8 | 42 | 1.7 |
| hiera_large | L40S | 63.4 | 66.2 | 1.67 |
| hiera_large | A100-PCIE-40GB | 47.9 | 48.5 | 1.67 |
| hiera_large | RTX 3090 | 69.5 | 70.2 | 1.67 |
| hiera_large | GB10 | 92 | 92.8 | 1.69 |
| hiera_large | RTX PRO 6000 Blackwell Server Edition | 506 | 603 | 1.7 |

### track (6 rows)

| gpu | tracker | fps | unique_ids | vram_gb |
|---|---|---|---|---|
| RTX 4090 | bytetrack.yaml | 2.2 | 4 | 24 |
| RTX 4090 | botsort.yaml | 10.9 | 4 | 24 |
| RTX 4090 | bytetrack.yaml | 62.6 | 3 | 24 |
| RTX 4090 | bytetrack.yaml | 49.8 | 4 | 24 |
| RTX 4090 | botsort.yaml | 30.7 | 3 | 24 |
| RTX 4090 | botsort.yaml | 27.9 | 4 | 24 |

### yolo (12 rows)

| model | gpu | imgsz | input | batch | p50_ms | p95_ms | fps_effective | vram_gb |
|---|---|---|---|---|---|---|---|---|
| yolo11n | RTX 4090 | 640 |  | 1 | 17.3 | 23.8 | 57.9 | 0.04 |
| yolo11n | RTX 4090 | 640 |  | 8 | 68.1 | 73.5 | 117 | 0.13 |
| yolo11n | RTX 4090 | 1280 | 1080p | 1 | 18.9 | 22.2 |  | 24 |
| yolo11s | RTX 4090 | 640 |  | 1 | 19.6 | 23.4 | 51 | 0.07 |
| yolo11s | RTX 4090 | 640 |  | 8 | 67.9 | 73.9 | 118 | 0.22 |
| yolo11s | RTX 4090 | 1280 | 1080p | 1 | 19.1 | 24.5 |  | 24 |
| yolo11m | RTX 4090 | 640 |  | 1 | 21.6 | 28.6 | 46.3 | 0.13 |
| yolo11m | RTX 4090 | 640 |  | 8 | 67.2 | 74.6 | 119 | 0.41 |
| yolo11m | RTX 4090 | 1280 | 1080p | 1 | 21.7 | 26.5 |  | 24 |
| yolo11x | RTX 4090 | 640 |  | 1 | 30.3 | 38.6 | 33 | 0.29 |
| yolo11x | RTX 4090 | 640 |  | 8 | 81 | 93.9 | 98.8 | 0.71 |
| yolo11x | RTX 4090 | 1280 | 1080p | 1 | 27.9 | 37 |  | 24 |

### yolo_gpu_only (120 rows)

| model | gpu | imgsz | batch | p50_ms | p95_ms | fps_gpu | vram_gb |
|---|---|---|---|---|---|---|---|
| yolo11n | RTX 4090 | 640 | 1 | 16.3 | 18.9 | 61.4 | 0.05 |
| yolo11n | RTX 4090 | 640 | 8 | 16.8 | 18.9 | 476 | 0.15 |
| yolo11n | RTX 4090 | 640 | 16 | 16.4 | 16.7 | 976 | 0.27 |
| yolo11n | RTX 4090 | 1280 | 1 | 16.1 | 16.4 | 62 | 0.09 |
| yolo11n | RTX 4090 | 1280 | 4 | 16 | 16.5 | 251 | 0.27 |
| yolo11m | RTX 4090 | 640 | 1 | 21.4 | 21.9 | 46.6 | 0.12 |
| yolo11m | RTX 4090 | 640 | 8 | 21.2 | 21.9 | 377 | 0.48 |
| yolo11m | RTX 4090 | 640 | 16 | 21.8 | 21.8 | 734 | 0.89 |
| yolo11m | RTX 4090 | 1280 | 1 | 20.6 | 22.1 | 48.5 | 0.28 |
| yolo11m | RTX 4090 | 1280 | 4 | 22.8 | 22.8 | 176 | 0.89 |
| yolo11x | RTX 4090 | 640 | 1 | 33.8 | 34.7 | 29.6 | 0.22 |
| yolo11x | RTX 4090 | 640 | 8 | 33.1 | 33.6 | 242 | 0.82 |
| yolo11x | RTX 4090 | 640 | 16 | 49.6 | 49.7 | 322 | 1.5 |
| yolo11x | RTX 4090 | 1280 | 1 | 31.1 | 31.7 | 32.2 | 0.48 |
| yolo11x | RTX 4090 | 1280 | 4 | 53.7 | 53.8 | 74.5 | 1.5 |
| yolo11n | RTX PRO 6000 Blackwell Server Edition | 640 | 1 | 9.22 | 9.3 | 108 | 0.05 |
| yolo11n | RTX PRO 6000 Blackwell Server Edition | 640 | 8 | 9.4 | 9.44 | 851 | 0.15 |
| yolo11n | RTX PRO 6000 Blackwell Server Edition | 640 | 16 | 9.45 | 9.67 | 1.69e+03 | 0.27 |
| yolo11n | RTX PRO 6000 Blackwell Server Edition | 1280 | 1 | 9.27 | 9.32 | 108 | 0.09 |
| yolo11n | RTX PRO 6000 Blackwell Server Edition | 1280 | 4 | 9.32 | 9.39 | 429 | 0.27 |
| yolo11m | RTX PRO 6000 Blackwell Server Edition | 640 | 1 | 11.8 | 11.8 | 84.7 | 0.12 |
| yolo11m | RTX PRO 6000 Blackwell Server Edition | 640 | 8 | 11.8 | 11.9 | 675 | 0.48 |
| yolo11m | RTX PRO 6000 Blackwell Server Edition | 640 | 16 | 14.1 | 14.1 | 1.14e+03 | 0.89 |
| yolo11m | RTX PRO 6000 Blackwell Server Edition | 1280 | 1 | 11.8 | 11.9 | 84.5 | 0.28 |
| yolo11m | RTX PRO 6000 Blackwell Server Edition | 1280 | 4 | 14.3 | 14.3 | 280 | 0.89 |
| yolo11x | RTX PRO 6000 Blackwell Server Edition | 640 | 1 | 18.3 | 18.4 | 54.6 | 0.22 |
| yolo11x | RTX PRO 6000 Blackwell Server Edition | 640 | 8 | 18.1 | 18.2 | 441 | 0.82 |
| yolo11x | RTX PRO 6000 Blackwell Server Edition | 640 | 16 | 31.4 | 31.6 | 509 | 1.49 |
| yolo11x | RTX PRO 6000 Blackwell Server Edition | 1280 | 1 | 17.9 | 18 | 55.9 | 0.48 |
| yolo11x | RTX PRO 6000 Blackwell Server Edition | 1280 | 4 | 31.2 | 31.4 | 128 | 1.49 |
| yolo11n | RTX 5090 | 640 | 1 | 3.59 | 4.82 | 279 | 0.05 |
| yolo11n | RTX 5090 | 640 | 8 | 3.63 | 4.29 | 2.2e+03 | 0.15 |
| yolo11n | RTX 5090 | 640 | 16 | 3.31 | 4.79 | 4.83e+03 | 0.27 |
| yolo11n | RTX 5090 | 1280 | 1 | 3.31 | 5.27 | 302 | 0.09 |
| yolo11n | RTX 5090 | 1280 | 4 | 3.75 | 5.25 | 1.07e+03 | 0.27 |
| yolo11m | RTX 5090 | 640 | 1 | 4.24 | 6.32 | 236 | 0.12 |
| yolo11m | RTX 5090 | 640 | 8 | 7.48 | 7.49 | 1.07e+03 | 0.48 |
| yolo11m | RTX 5090 | 640 | 16 | 14.7 | 14.7 | 1.09e+03 | 0.89 |
| yolo11m | RTX 5090 | 1280 | 1 | 4.63 | 6.77 | 216 | 0.28 |
| yolo11m | RTX 5090 | 1280 | 4 | 14.9 | 14.9 | 269 | 0.89 |
| yolo11x | RTX 5090 | 640 | 1 | 6.42 | 9.03 | 156 | 0.22 |
| yolo11x | RTX 5090 | 640 | 8 | 18.2 | 18.2 | 440 | 0.82 |
| yolo11x | RTX 5090 | 640 | 16 | 34.1 | 34.2 | 469 | 1.49 |
| yolo11x | RTX 5090 | 1280 | 1 | 9.61 | 9.62 | 104 | 0.48 |
| yolo11x | RTX 5090 | 1280 | 4 | 34.8 | 34.8 | 115 | 1.49 |
| yolo11n | H100 NVL | 640 | 1 | 6.35 | 6.41 | 158 | 0.05 |
| yolo11n | H100 NVL | 640 | 8 | 6.57 | 6.63 | 1.22e+03 | 0.15 |
| yolo11n | H100 NVL | 640 | 16 | 6.57 | 6.61 | 2.44e+03 | 0.27 |
| yolo11n | H100 NVL | 1280 | 1 | 6.4 | 6.46 | 156 | 0.09 |
| yolo11n | H100 NVL | 1280 | 4 | 6.6 | 6.65 | 606 | 0.27 |
| yolo11m | H100 NVL | 640 | 1 | 8.13 | 8.23 | 123 | 0.12 |
| yolo11m | H100 NVL | 640 | 8 | 8.45 | 8.54 | 947 | 0.48 |
| yolo11m | H100 NVL | 640 | 16 | 13.8 | 13.9 | 1.16e+03 | 0.89 |
| yolo11m | H100 NVL | 1280 | 1 | 8.17 | 8.22 | 122 | 0.28 |
| yolo11m | H100 NVL | 1280 | 4 | 14.3 | 14.4 | 279 | 0.89 |
| yolo11x | H100 NVL | 640 | 1 | 12.6 | 12.7 | 79.2 | 0.22 |
| yolo11x | H100 NVL | 640 | 8 | 16.6 | 16.7 | 480 | 0.82 |
| yolo11x | H100 NVL | 640 | 16 | 30.7 | 31.7 | 521 | 1.49 |
| yolo11x | H100 NVL | 1280 | 1 | 12.5 | 12.7 | 79.9 | 0.48 |
| yolo11x | H100 NVL | 1280 | 4 | 31 | 32.1 | 129 | 1.49 |
| yolo11n | L40S | 640 | 1 | 4.48 | 4.63 | 223 | 0.05 |
| yolo11n | L40S | 640 | 8 | 7.96 | 8.24 | 1e+03 | 0.15 |
| yolo11n | L40S | 640 | 16 | 7.98 | 8.35 | 2e+03 | 0.27 |
| yolo11n | L40S | 1280 | 1 | 8.12 | 8.34 | 123 | 0.09 |
| yolo11n | L40S | 1280 | 4 | 5.23 | 5.24 | 765 | 0.27 |
| yolo11m | L40S | 640 | 1 | 9.9 | 10.2 | 101 | 0.12 |
| yolo11m | L40S | 640 | 8 | 10.8 | 10.8 | 744 | 0.48 |
| yolo11m | L40S | 640 | 16 | 23.1 | 23.2 | 693 | 0.89 |
| yolo11m | L40S | 1280 | 1 | 9.62 | 9.74 | 104 | 0.28 |
| yolo11m | L40S | 1280 | 4 | 23.7 | 23.8 | 169 | 0.89 |
| yolo11x | L40S | 640 | 1 | 15.1 | 15.2 | 66.3 | 0.22 |
| yolo11x | L40S | 640 | 8 | 24.4 | 24.4 | 328 | 0.82 |
| yolo11x | L40S | 640 | 16 | 51.5 | 51.6 | 310 | 1.5 |
| yolo11x | L40S | 1280 | 1 | 14.9 | 15 | 67.2 | 0.48 |
| yolo11x | L40S | 1280 | 4 | 55.2 | 55.3 | 72.5 | 1.5 |
| yolo11n | A100-PCIE-40GB | 640 | 1 | 9.56 | 9.98 | 105 | 0.05 |
| yolo11n | A100-PCIE-40GB | 640 | 8 | 9.37 | 9.79 | 854 | 0.15 |
| yolo11n | A100-PCIE-40GB | 640 | 16 | 9.44 | 9.8 | 1.69e+03 | 0.27 |
| yolo11n | A100-PCIE-40GB | 1280 | 1 | 9.28 | 9.34 | 108 | 0.09 |
| yolo11n | A100-PCIE-40GB | 1280 | 4 | 9.26 | 9.64 | 432 | 0.27 |
| yolo11m | A100-PCIE-40GB | 640 | 1 | 13 | 13.5 | 76.7 | 0.12 |
| yolo11m | A100-PCIE-40GB | 640 | 8 | 14.3 | 17.9 | 561 | 0.48 |
| yolo11m | A100-PCIE-40GB | 640 | 16 | 26.2 | 26.5 | 611 | 0.89 |
| yolo11m | A100-PCIE-40GB | 1280 | 1 | 11.8 | 12.7 | 84.7 | 0.28 |
| yolo11m | A100-PCIE-40GB | 1280 | 4 | 26.9 | 27.4 | 148 | 0.89 |
| yolo11x | A100-PCIE-40GB | 640 | 1 | 20.9 | 23.4 | 47.9 | 0.27 |
| yolo11x | A100-PCIE-40GB | 640 | 8 | 31.7 | 32.2 | 252 | 1.18 |
| yolo11x | A100-PCIE-40GB | 640 | 16 | 57.7 | 58.3 | 277 | 1.79 |
| yolo11x | A100-PCIE-40GB | 1280 | 1 | 18.8 | 19.3 | 53.2 | 0.66 |
| yolo11x | A100-PCIE-40GB | 1280 | 4 | 62 | 62.5 | 64.5 | 2.23 |
| yolo11n | RTX 3090 | 640 | 1 | 10.1 | 10.2 | 98.6 | 0.05 |
| yolo11n | RTX 3090 | 640 | 8 | 10.3 | 10.7 | 779 | 0.15 |
| yolo11n | RTX 3090 | 640 | 16 | 10.8 | 12.3 | 1.48e+03 | 0.27 |
| yolo11n | RTX 3090 | 1280 | 1 | 10.3 | 10.3 | 97.4 | 0.09 |
| yolo11n | RTX 3090 | 1280 | 4 | 11.8 | 11.9 | 338 | 0.27 |
| yolo11m | RTX 3090 | 640 | 1 | 13.2 | 13.6 | 75.7 | 0.12 |
| yolo11m | RTX 3090 | 640 | 8 | 21.2 | 21.4 | 376 | 0.48 |
| yolo11m | RTX 3090 | 640 | 16 | 40.3 | 40.7 | 397 | 0.89 |
| yolo11m | RTX 3090 | 1280 | 1 | 13.1 | 13.1 | 76.6 | 0.28 |
| yolo11m | RTX 3090 | 1280 | 4 | 41.4 | 41.5 | 96.7 | 0.89 |
| yolo11x | RTX 3090 | 640 | 1 | 23.1 | 24.5 | 43.4 | 0.22 |
| yolo11x | RTX 3090 | 640 | 8 | 50 | 50.3 | 160 | 0.82 |
| yolo11x | RTX 3090 | 640 | 16 | 96.9 | 97.3 | 165 | 1.5 |
| yolo11x | RTX 3090 | 1280 | 1 | 28.2 | 28.3 | 35.5 | 0.48 |
| yolo11x | RTX 3090 | 1280 | 4 | 100 | 101 | 40 | 1.5 |
| yolo11n | GB10 | 640 | 1 | 3.26 | 3.38 | 307 | 0.05 |
| yolo11n | GB10 | 640 | 8 | 9.67 | 10.5 | 827 | 0.15 |
| yolo11n | GB10 | 640 | 16 | 19.9 | 20.8 | 802 | 0.27 |
| yolo11n | GB10 | 1280 | 1 | 5.54 | 6.03 | 180 | 0.09 |
| yolo11n | GB10 | 1280 | 4 | 20.6 | 21.5 | 194 | 0.27 |
| yolo11m | GB10 | 640 | 1 | 6.42 | 7.18 | 156 | 0.12 |
| yolo11m | GB10 | 640 | 8 | 45.7 | 46.4 | 175 | 0.48 |
| yolo11m | GB10 | 640 | 16 | 93.3 | 94.6 | 172 | 0.89 |
| yolo11m | GB10 | 1280 | 1 | 20.5 | 21.2 | 48.7 | 0.28 |
| yolo11m | GB10 | 1280 | 4 | 95.1 | 96.6 | 42.1 | 0.89 |
| yolo11x | GB10 | 640 | 1 | 12 | 12.8 | 83.1 | 0.22 |
| yolo11x | GB10 | 640 | 8 | 94 | 95.2 | 85.1 | 0.82 |
| yolo11x | GB10 | 640 | 16 | 195 | 197 | 81.9 | 1.49 |
| yolo11x | GB10 | 1280 | 1 | 45.4 | 46.1 | 22 | 0.48 |
| yolo11x | GB10 | 1280 | 4 | 202 | 203 | 19.8 | 1.49 |


## What this cost

Whole campaign: US$ 50.79 (about R$ 249), shared by all repositories of this series. Rental share of the GPUs used in this repo:

| GPU | rentals | hours | rental cost (USD) |
|---|---|---|---|
| GB10 | 3 | 0.0 | 0.0 |
| L40S | 10 | 4.91 | 3.92 |
| RTX 3090 | 30 | 11.8 | 2.97 |
| RTX 4090 | 29 | 6.22 | 4.05 |
| RTX 5090 | 39 | 12.44 | 8.54 |

Rental figures are hourly price x time of each rented instance; marketplace download fees and idle time are in the campaign total, not in this table.

## Reproduce

`bash scripts/run.sh` documents the exact commands used on the rented instance (Ubuntu, CUDA, Python venv). Evidence files are the unmodified measured rows (host location and reliability removed).
