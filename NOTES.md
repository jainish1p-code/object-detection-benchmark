# Project handoff notes

## Goal
Authoritative assignment reference: Track A, at least 3 model configurations from
at least 2 detector families, at least 2 datasets/domains, reproducible measurements,
and at least 2 technical analyses. Tutorial starting point:
https://github.com/lkk688/DeepDataMiningLearning/blob/main/DeepDataMiningLearning/ngdet/TUTORIAL.md
Our selected analyses are cross-dataset generalization and speed-accuracy tradeoff.

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
  dataset category ids, scores with pycocotools, saves aggregate and per-class AP,
  settings, environment, exact command, and selected image IDs to JSON.
  Run from repo root: PYTHONPATH=src python src/eval_coco.py --model frcnn
- Score threshold 0.001 for mAP (low on purpose; mAP ranks detections). Each family keeps
  its own NMS default (frcnn 0.5, YOLO 0.7, DETR none); recorded in the results json.
- src/benchmark.py: batch-size-1 predict() latency, excluding file loading and warmup;
  CUDA synchronization, mean/p50/p95, FPS, peak allocated/reserved CUDA memory,
  raw timings and reproducibility metadata. CPU smoke checks passed for all 3 models;
  Faster R-CNN GPU smoke benchmark passed on a V100-32: mean 62.53 ms, p50 62.44,
  p95 66.08, FPS 15.99, peak allocated 1126.19 MiB, reserved 1474.00 MiB
  (5 images, 5 warmup, 10 measured predictions; preliminary only).
- scripts/run_coco.slurm: submit from the cluster repo root with sbatch; requests
  one V100-32 for up to 2 hours, evaluates all 5,000 images and benchmarks each model
  in its own process, saves results/coco/<jobid>/ and coco_<jobid>.log.
- src/prepare_kitti.py converts labels to COCO-format GT; unified classes vehicle/person
  (Car, Van, Truck, Tram -> vehicle; Pedestrian, Person_sitting -> person; Cyclist,
  Misc, DontCare excluded with an explicit ignore policy). For evaluation without
  retraining, record a deterministic labeled subset; a train/val split is not needed.
  If fine-tuning is added, prevent sequence overlap using verified sequence metadata;
  do not assume adjacent KITTI object-detection image IDs are consecutive frames.
- src/taxonomy.py and eval_coco.py --taxonomy vehicle_person apply the same label
  definitions to ground truth and detector outputs. COCO car/truck/bus/train -> vehicle;
  person -> person. All images, including those without mapped objects, are retained.
  KITTI Cyclist/Misc/DontCare boxes become COCO crowd-ignore annotations for BOTH
  categories. COCO matching prefers real GT, then ignores detections overlapping these
  regions using intersection/detection-area at each IoU threshold (.50:.95).
  No KITTI difficulty filtering; boxes clipped to image bounds; area from box size.
  This is a documented custom COCO-style protocol, not official KITTI AP_R40.
  Label definitions, annotation conventions and object distributions still differ;
  differences in shared-class AP are not a pure measurement of domain shift alone.
- scripts/run_domains.slurm prepares KITTI, saves GT sanity scores, evaluates all
  models on full COCO and KITTI using the shared taxonomy, then benchmarks KITTI.
  Outputs: results/domains/<jobid>/ and domains_<jobid>.log. Job 50151110 reached
  the completion marker; all expected JSONs and the full log are backed up locally.

## Status
- Laptop (macOS, .venv, Python 3.11.5): all 3 models run on a 20-image COCO mini set.
  AP: frcnn 0.619, yolo11s 0.602, detr 0.535 (noisy, tiny sample; sanity check with
  GT-as-predictions gives AP 1.000).
- Bridges-2 (PSC, user jpatel17, allocation cis260281p, project dir
  /ocean/projects/cis260281p/jpatel17): conda env at $PROJECT/envs/det (Python 3.11.18)
  now has torch 2.7.1+cu118, torchvision 0.22.1+cu118, CUDA build 11.8, and remaining
  requirements installed. pip check and all detector imports passed. V100-32 GPU
  session on v003 confirmed: CUDA tensor operation and all three detectors passed.
  Five-image AP: frcnn 0.630, yolo11s 0.527, detr 0.466 (smoke results only).
- Storage cleanup: user removed re-downloadable DriveLM datasets and hf_cache;
  last reported free quota was 24 GB before the remaining library installation.
  DriveLM code/environments were kept. COCO val2017 extracted in the cluster repo's
  data/coco/; verified 5,000 annotation-listed images and zero missing files.
  User transferred the updated eval_coco.py and benchmark.py to the cluster.
- tmux session det-setup was created on bridges2-login012; reconnect from within
  Bridges-2 via ssh jpatel17@br012.ib.bridges2.psc.edu before attaching.
- Repo on cluster: $PROJECT/object-detection-benchmark. Rule: push from laptop only,
  pull on the cluster.
- Storage update: additional allocation approved; user reports 50 GB available in
  Ocean. Keep COCO, KITTI, archives, and results until the entire homework is finished.
  Do not delete anything as part of the remaining work without explicit approval.
- KITTI downloaded and uploaded; labeled training extraction verified by user:
  7,481 images, 7,481 labels, zero unmatched files in data/kitti/training/.
  ZIPs kept at /ocean/projects/cis260281p/jpatel17/. Source URLs:
  https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
  https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip

## Full native-COCO results (reported from Bridges-2)
Job 50149631: COMPLETED, exit 0:0, elapsed 00:15:25, V100-SXM2-32GB on v003.
All models evaluated on all 5,000 COCO val2017 images. Benchmarks used 100 selected
images, 20 warmup predictions, and 100 measured predictions, batch size 1.
Settings: score_thr=0.001, family-specific default NMS and input resizing.

| Model | AP | AP50 | AP75 | Mean latency (ms) | FPS | Peak allocated (MiB) |
| --- | --- | --- | --- | --- | --- | --- |
| frcnn | 0.469 | 0.679 | 0.514 | 64.16 | 15.59 | 1132.92 |
| yolo11s | 0.455 | 0.620 | 0.492 | 11.79 | 84.82 | 110.55 |
| detr | 0.420 | 0.623 | 0.442 | 38.73 | 25.82 | 472.02 |

Artifacts on cluster: results/coco/50149631/ (3 evaluation JSONs, 3 benchmark JSONs,
packages.txt), plus coco_50149631.log at repo root. Evaluation and benchmark JSONs
are now present locally in results/50149631/. Inspected all three evaluations:
each has 5,000 images, 80 per-class AP entries, exact command and detector settings.
Exact p50/p95, AP by size, AR, per-class AP and environment fields
remain in those JSONs. Preserve all artifacts and COCO images for the shared-taxonomy run.
Observed speed-accuracy tradeoff: YOLO11s has about 5.4x Faster R-CNN's FPS with AP
lower by 0.014 (1.4 percentage points), under these model-specific default resolutions.

## Shared COCO/KITTI results and report
Domain job 50151110 finished according to the completion marker in its log on v020.
Local backup: results/50151110/ and results/domains_50151110.log.
All 6 shared-class evaluation JSONs and 3 KITTI benchmarks are present.
Both full ground-truth sanity runs have aggregate and per-class AP 1.0000.
KITTI medium-size sanity AP rounds to 0.9999; this small discrepancy is disclosed.

| Model | Shared COCO AP | KITTI AP | AP drop (pp) | Vehicle drop (pp) | Person drop (pp) |
| --- | --- | --- | --- | --- | --- |
| frcnn | 0.5731 | 0.3962 | 17.69 | 6.20 | 29.18 |
| yolo11s | 0.5564 | 0.3648 | 19.16 | 6.48 | 31.85 |
| detr | 0.4991 | 0.3545 | 14.46 | 2.80 | 26.13 |

KITTI mean latency/FPS/allocated MiB: frcnn 53.33/18.75/1088.96,
yolo11s 8.86/112.92/82.84, detr 27.45/36.43/321.54.
Person AP degrades much more than vehicle AP. Medium/large AP have larger absolute
cross-dataset drops than small AP; small AP is still lowest within KITTI.

scripts/make_report.py validates 9 evaluations, 6 benchmarks, and 2 sanity runs,
then generates README.md, results/summary/{accuracy,per_class_ap,benchmarks,generalization}.csv,
commands.md (17 exact recorded commands plus conversion), coco_per_class.md (80 classes),
source_sha256.json, and three figures in PNG/PDF under results/summary/figures/.
The README includes both required analyses, model/dataset settings, AP/AR tables,
per-class results, timing/memory, reproducibility instructions, and limitations.
No extra GPU runs are needed for the current report.

## Final submission requirements and artifacts
User supplied additional requirements: one PDF, either a 6–10 page technical report
or 15–25 slides, plus code/configurations/command log. Track A needs failure analysis,
conclusions and at least 8 credible references, with at least 4 research papers and
2 official repository/framework sources. Student must verify AI-generated work.
User explicitly chose to omit author name and course title from the PDF.

scripts/build_submission.py generates a nine-page submission/technical_report.pdf,
its Markdown source, and submission/experiment_code_and_results.zip. The ZIP contains
source, Slurm scripts, raw JSONs, package snapshots, both real job logs, CSVs, figures,
references and command records. It excludes datasets, weights and smoke outputs.
report/references.json contains 12 verified primary references: 6 original research
papers, 5 official framework/repository sources, and the course tutorial. The PDF
cites them in its body and includes an AI-use disclosure. Its dedicated failure
analysis is quantitative (class/scale/localization/recall/tail latency); no per-image
prediction files or qualitative failure screenshots were collected.

Local validation: four synthetic domain tests passed; native and shared mini-COCO
GT sanity AP 1.000; report checks saved counts, IDs, category counts, timing percentiles,
FPS against raw samples, and full-dataset sanity scores. Shell syntax checks passed.

## Next steps
1. Review submission/technical_report.pdf for understanding and accuracy.
2. Submit the PDF and upload/link experiment_code_and_results.zip as the supporting
   code/configuration/evidence artifact. No remote submission has been performed.
   GitHub sync requested for https://github.com/jainish1p-code/object-detection-benchmark;
   source, report and small experiment artifacts are included; see git history for status.
3. Keep datasets and archives. Never commit data, weights or results/smoke/.

## Gotchas hit so far
- macOS Python needs SSL_CERT_FILE=certifi path for weight downloads.
- Commit nothing large: data/, *.pt, results/smoke/ are in .gitignore.
- A dropped SSH connection corrupted the conda cache; use tmux for long commands.
- Quota banner "6 / 100 GB" means 6 GB remaining.
- data.bridges2.psc.edu rejects interactive shell login ("Only file transfers are
  allowed"). Use scp/sftp from the Mac. Downloads were performed on the Mac and
  uploaded through the DTN; do not suggest tmux/SSH-shell downloads on the DTN.
