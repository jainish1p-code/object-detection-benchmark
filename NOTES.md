# Project handoff notes

## Goal
Homework: 2D object detection benchmark. Evaluate 3 models from 3 families
(Faster R-CNN, YOLO11, DETR) on 2 datasets (COCO val2017, KITTI) without retraining,
plus analysis: cross-dataset generalization and speed-accuracy tradeoff
(fine-tuning YOLO on KITTI is optional). Metrics: AP, AP50, AP75, AP small/medium/large,
per-class AP, AR, latency/FPS, GPU memory. Every experiment must record model/checkpoint,
dataset/split, input resolution, conf/NMS thresholds, hardware, framework versions,
and the exact command. Deadline: Nov 2 [time].

## Design
- src/detectors.py: Detector interface. predict(PIL img) -> (boxes_xyxy, scores, class NAMES).
  Detectors: frcnn (torchvision fasterrcnn_resnet50_fpn_v2, COCO_V1), yolo11s (Ultralytics),
  detr (facebook/detr-resnet-50, HF).
- src/eval_coco.py: runs a detector over a COCO-format dataset, maps class names to
  dataset category ids, scores with pycocotools, saves a results json with settings + env.
  Run from repo root: PYTHONPATH=src python src/eval_coco.py --model frcnn
- Score threshold 0.001 for mAP (low on purpose; mAP ranks detections). Each family keeps
  its own NMS default (frcnn 0.5, YOLO 0.7, DETR none); recorded in the results json.
- Plan for KITTI: convert labels to COCO-format GT; unified classes vehicle/person
  (Car, Van, Truck, Tram -> vehicle; Pedestrian, Person_sitting -> person; Cyclist,
  Misc, DontCare ignored). Split train/val by contiguous frame blocks (no random split,
  frames are near-duplicates).

## Status
- Laptop (macOS, .venv, Python 3.11.5): all 3 models run on a 20-image COCO mini set.
  AP: frcnn 0.619, yolo11s 0.602, detr 0.535 (noisy, tiny sample; sanity check with
  GT-as-predictions gives AP 1.000).
- Bridges-2 (PSC, user jpatel17, allocation cis260281p, project dir
  /ocean/projects/cis260281p/jpatel17): conda env at $PROJECT/envs/det (Python 3.11.18)
  created but torch and the other libraries are NOT installed yet. Waiting on a
  storage quota increase; cleaning old data to free space.
- Repo on cluster: $PROJECT/object-detection-benchmark. Rule: push from laptop only,
  pull on the cluster.
- KITTI access: [approved / pending].

## Next steps
1. Install libs on Bridges-2 (pip install --no-cache-dir ...), inside tmux.
2. Download COCO val2017 to $PROJECT/data/coco.
3. Short GPU session; full COCO eval of 3 models.
4. KITTI subset + converter + eval.
5. src/benchmark.py: latency (warmup, cuda synchronize, p50/p95), peak memory.
6. README with results tables, reproducibility table, analysis.

## Gotchas hit so far
- macOS Python needs SSL_CERT_FILE=certifi path for weight downloads.
- Commit nothing large: data/, *.pt, results/smoke/ are in .gitignore.
- A dropped SSH connection corrupted the conda cache; use tmux for long commands.
- Quota banner "6 / 100 GB" means 6 GB remaining.