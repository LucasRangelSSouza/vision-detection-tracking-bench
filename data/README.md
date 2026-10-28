# Data

Curated CSV files from the campaign (public Kaggle dataset `lucasrangelss/gpu-ai-benchmark-campaign-2026`). One row per measured run or attempt; the columns are described in the dataset page. Host locations and instance identifiers were removed.

- `libs_runs.csv`: every timed run of the per-card library sweeps: YOLO11 grid, trackers, SAM 2, Grounding DINO, Mask R-CNN, plus generation, speech and encoding rows used by other repositories
- `segmentation_attempts.csv`: all attempts at SAM 2, Grounding DINO, Mask R-CNN and Detectron2 with status and failure reason
