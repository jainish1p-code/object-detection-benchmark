"""Validate saved experiments and regenerate README, CSV tables, and figures."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "detection-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = ("frcnn", "yolo11s", "detr")
NAMES = {"frcnn": "Faster R-CNN", "yolo11s": "YOLO11s", "detr": "DETR"}
METRICS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
           "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large")


def read(path):
    return json.loads(path.read_text())


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
    ])


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, default=Path("results/50149631"))
    parser.add_argument("--domains-dir", type=Path, default=Path("results/50151110"))
    args = parser.parse_args()
    out = Path("results/summary")
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    evaluations, benchmarks, sources = {}, {}, []
    for model in MODELS:
        paths = {
            "COCO-80": args.native_dir / f"{model}_eval.json",
            "COCO-2": args.domains_dir / f"coco_{model}_eval.json",
            "KITTI-2": args.domains_dir / f"kitti_{model}_eval.json",
        }
        for dataset, path in paths.items():
            record = read(path)
            expected = 7481 if dataset == "KITTI-2" else 5000
            if record["num_images"] != expected or len(set(record["image_ids"])) != expected:
                raise ValueError(f"{path}: incomplete or duplicated image IDs")
            if len(record["per_class_ap"]) != (80 if dataset == "COCO-80" else 2):
                raise ValueError(f"{path}: wrong class count")
            if not all(math.isfinite(record["stats"][key]) and 0 <= record["stats"][key] <= 1
                       for key in METRICS):
                raise ValueError(f"{path}: invalid metrics")
            if not record.get("command") or not record.get("detector"):
                raise ValueError(f"{path}: missing reproducibility metadata")
            evaluations[dataset, model] = record
            sources.append(path)
        for dataset, path in {
            "COCO": args.native_dir / f"{model}_benchmark.json",
            "KITTI": args.domains_dir / f"kitti_{model}_benchmark.json",
        }.items():
            record = read(path)
            samples = record["samples_ms"]
            if len(samples) != 100 or record["args"]["warmup"] != 20:
                raise ValueError(f"{path}: unexpected timing protocol")
            if record["num_images"] != 100 or not record["protocol"]["cuda_synchronize"]:
                raise ValueError(f"{path}: unexpected benchmark images or synchronization")
            if not all(math.isfinite(sample) and sample > 0 for sample in samples):
                raise ValueError(f"{path}: invalid latency samples")
            for key, value in zip(("mean", "p50", "p95"),
                                  (np.mean(samples), *np.percentile(samples, [50, 95]))):
                if not math.isclose(record["latency_ms"][key], value, rel_tol=1e-9):
                    raise ValueError(f"{path}: inconsistent {key}")
            if not math.isclose(record["fps"], 1000 / np.mean(samples), rel_tol=1e-9):
                raise ValueError(f"{path}: inconsistent FPS")
            benchmarks[dataset, model] = record
            sources.append(path)
        if evaluations["COCO-80", model]["image_ids"] != evaluations["COCO-2", model]["image_ids"]:
            raise ValueError("native and shared COCO runs selected different images")

    sanity_rows = []
    for dataset in ("coco", "kitti"):
        path = args.domains_dir / f"{dataset}_sanity.json"
        record = read(path)
        if record["stats"]["AP"] != 1.0 or any(v != 1.0 for v in record["per_class_ap"].values()):
            raise ValueError(f"{path}: failed ground-truth sanity check")
        sanity_rows.append([dataset.upper(), record["num_images"], "1.0000", "1.0000"])
        sources.append(path)

    # Every model must use identical evaluation images within a domain.
    for dataset in ("COCO-80", "COCO-2", "KITTI-2"):
        baseline = evaluations[dataset, MODELS[0]]
        for model in MODELS:
            r = evaluations[dataset, model]
            if r["image_ids"] != baseline["image_ids"]:
                raise ValueError(f"{dataset}: models evaluated different image subsets")
            if r["args"]["score_thr"] != 0.001:
                raise ValueError(f"{dataset}: unexpected confidence threshold")

    metric_rows, class_rows, timing_rows, gap_rows = [], [], [], []
    for (dataset, model), r in evaluations.items():
        metric_rows.append({"dataset": dataset, "model": model, "images": r["num_images"], **r["stats"]})
        for name, value in r["per_class_ap"].items():
            class_rows.append({"dataset": dataset, "model": model, "class": name, "AP": value})
    for (dataset, model), r in benchmarks.items():
        timing_rows.append({"dataset": dataset, "model": model,
                            **{f"{key}_ms": value for key, value in r["latency_ms"].items()},
                            "fps": r["fps"], **r["gpu_memory"]})
    for model in MODELS:
        coco, kitti = evaluations["COCO-2", model], evaluations["KITTI-2", model]
        gap_rows.append({"model": model, "COCO_AP": coco["stats"]["AP"],
                         "KITTI_AP": kitti["stats"]["AP"],
                         "AP_drop_pp": 100 * (coco["stats"]["AP"] - kitti["stats"]["AP"]),
                         **{f"{name}_drop_pp": 100 * (coco["per_class_ap"][name] - kitti["per_class_ap"][name])
                            for name in ("vehicle", "person")}})
    for name, rows in (("accuracy.csv", metric_rows), ("per_class_ap.csv", class_rows),
                       ("benchmarks.csv", timing_rows), ("generalization.csv", gap_rows)):
        write_csv(out / name, rows)

    commands = ["# Recorded experiment commands", "", "Run from the cluster repository root.",
                "These are the original commands, including their original output paths.", ""]
    for path in sources:
        r = read(path)
        commands.extend([f"## {path}", "", "```bash", r["command"], "```", ""])
    conversion = evaluations["KITTI-2", "frcnn"]["dataset_info"]["conversion_command"]
    commands.extend(["## KITTI annotation conversion", "", "```bash", conversion, "```", ""])
    (out / "commands.md").write_text("\n".join(commands))
    for directory in (args.native_dir, args.domains_dir):
        sources.append(directory / "packages.txt")
    manifest = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(set(sources))
    }
    (out / "source_sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = ("#2563eb", "#dc6803", "#16836b")
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7, 4))
    for offset, dataset, label, color in ((-0.18, "COCO-2", "COCO (2 classes)", "#2563eb"),
                                         (0.18, "KITTI-2", "KITTI (2 classes)", "#dc6803")):
        bars = ax.bar(x + offset, [evaluations[dataset, m]["stats"]["AP"] for m in MODELS],
                      width=0.36, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set(xticks=x, xticklabels=[NAMES[m] for m in MODELS], ylabel="mAP@[.50:.95]",
           ylim=(0, 0.72), title="COCO to KITTI: shared vehicle/person taxonomy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "domain_accuracy.png", dpi=180)
    fig.savefig(figures / "domain_accuracy.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, dataset, metric_dataset in zip(axes, ("COCO", "KITTI"), ("COCO-80", "KITTI-2")):
        for model, color in zip(MODELS, colors):
            b = benchmarks[dataset, model]
            ap = evaluations[metric_dataset, model]["stats"]["AP"]
            ax.scatter(b["latency_ms"]["mean"], ap, color=color, s=85)
            ax.annotate(NAMES[model], (b["latency_ms"]["mean"], ap),
                        xytext=(4, 7), textcoords="offset points")
        ax.set(xlabel="Mean predict() latency (ms; lower is faster)", ylabel="mAP@[.50:.95]",
               title=f"{dataset}: {'80' if dataset == 'COCO' else '2'} classes", xlim=(0, 85),
               ylim=(0.30, 0.53) if dataset == "COCO" else (0.28, 0.47))
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "speed_accuracy.png", dpi=180)
    fig.savefig(figures / "speed_accuracy.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for offset, name, color in ((-0.18, "vehicle", "#2563eb"), (0.18, "person", "#dc6803")):
        bars = ax.bar(x + offset, [row[f"{name}_drop_pp"] for row in gap_rows],
                      width=0.36, label=name, color=color)
        ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set(xticks=x, xticklabels=[NAMES[m] for m in MODELS], ylabel="COCO AP − KITTI AP (percentage points)",
           ylim=(0, 39), title="Person AP drops more than vehicle AP")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "class_gap.png", dpi=180)
    fig.savefig(figures / "class_gap.pdf")
    plt.close(fig)

    accuracy_tables = []
    recall_tables = []
    for dataset in ("COCO-80", "COCO-2", "KITTI-2"):
        accuracy_tables.append(f"### {dataset}\n\n" + table(
            ["Model", "AP", "AP50", "AP75", "AP small", "AP medium", "AP large"],
            [[NAMES[m], *(f"{evaluations[dataset, m]['stats'][key]:.4f}" for key in METRICS[:6])]
             for m in MODELS]))
        recall_tables.append(f"### {dataset}\n\n" + table(
            ["Model", "AR1", "AR10", "AR100", "AR small", "AR medium", "AR large"],
            [[NAMES[m], *(f"{evaluations[dataset, m]['stats'][key]:.4f}" for key in METRICS[6:])]
             for m in MODELS]))
    gap_table = table(["Model", "COCO-2 AP", "KITTI-2 AP", "Drop (pp)", "Vehicle drop (pp)", "Person drop (pp)"],
                      [[NAMES[row["model"]], f"{row['COCO_AP']:.4f}", f"{row['KITTI_AP']:.4f}",
                        *(f"{row[key]:.2f}" for key in ("AP_drop_pp", "vehicle_drop_pp", "person_drop_pp"))]
                       for row in gap_rows])
    class_table = table(["Model", "COCO vehicle", "KITTI vehicle", "COCO person", "KITTI person"],
                        [[NAMES[m], *(f"{evaluations[d, m]['per_class_ap'][c]:.4f}"
                           for c in ("vehicle", "person") for d in ("COCO-2", "KITTI-2"))]
                         for m in MODELS])
    timing_table = table(["Dataset", "Model", "Mean ms", "p50 ms", "p95 ms", "FPS", "Allocated MiB", "Reserved MiB"],
                         [[r["dataset"], NAMES[r["model"]],
                           *(f"{r[key]:.2f}" for key in ("mean_ms", "p50_ms", "p95_ms", "fps",
                                                        "peak_allocated_mib", "peak_reserved_mib"))]
                          for r in timing_rows])
    scale_table = table(["Model", "Small AP drop (pp)", "Medium AP drop (pp)", "Large AP drop (pp)"],
                        [[NAMES[m], *(f"{100*(evaluations['COCO-2',m]['stats'][key]-evaluations['KITTI-2',m]['stats'][key]):.2f}"
                                     for key in ("AP_small", "AP_medium", "AP_large"))] for m in MODELS])
    env = benchmarks["COCO", "frcnn"]["env"]
    speedup = benchmarks["COCO", "yolo11s"]["fps"] / benchmarks["COCO", "frcnn"]["fps"]
    speedup_kitti = benchmarks["KITTI", "yolo11s"]["fps"] / benchmarks["KITTI", "frcnn"]["fps"]
    ap_cost = 100 * (evaluations["COCO-80", "frcnn"]["stats"]["AP"] - evaluations["COCO-80", "yolo11s"]["stats"]["AP"])
    native_class_table = table(["COCO class", "Faster R-CNN", "YOLO11s", "DETR"], [
        [name, *(f"{evaluations['COCO-80', m]['per_class_ap'][name]:.4f}" for m in MODELS)]
        for name in sorted(evaluations["COCO-80", "frcnn"]["per_class_ap"])
    ])
    (out / "coco_per_class.md").write_text("# Native COCO per-class AP\n\n" + native_class_table + "\n")
    accuracy_text = "\n\n".join(accuracy_tables)
    recall_text = "\n\n".join(recall_tables)
    readme = f"""# 2D Object Detection: COCO to KITTI

Track A experimental evaluation of three COCO-pretrained detectors from three families,
without retraining. The two primary analyses are cross-dataset generalization and the
speed–accuracy tradeoff. The implementation follows the detector/dataset/evaluator
separation in the [course tutorial](https://github.com/lkk688/DeepDataMiningLearning/blob/main/DeepDataMiningLearning/ngdet/TUTORIAL.md).

Submission artifacts: [nine-page technical report](submission/technical_report.pdf)
and [experiment code/results archive](submission/experiment_code_and_results.zip).
The PDF includes a dedicated quantitative failure analysis, AI-use disclosure, and
12 references (six original research papers and five official framework/repository sources).
Author and course title are omitted at the user's request.

## Main findings

Faster R-CNN has the highest overall AP on native COCO and on KITTI under the shared
taxonomy. YOLO11s provides {speedup:.2f}× its COCO FPS while losing {ap_cost:.2f} AP percentage
points; on KITTI it provides {speedup_kitti:.2f}× its FPS. DETR lies between them in latency and
allocated GPU memory, but has lower overall AP in these configurations.

All models lose AP on KITTI under a matched vehicle/person taxonomy. Person AP drops
much more than vehicle AP. This is evidence of weaker transfer to the driving dataset
under this protocol; dataset composition and annotation differences also contribute.

## Models and inference settings

| CLI name | Family / checkpoint | Input resizing | Confidence | NMS IoU | Maximum outputs |
| --- | --- | --- | --- | --- | --- |
| frcnn | Two-stage CNN; fasterrcnn_resnet50_fpn_v2, COCO_V1 | Short side 800, long side ≤1333 | 0.001 | 0.5 | 100 |
| yolo11s | One-stage CNN; yolo11s.pt | imgsz=640, aspect-preserving letterbox; actual rectangle depends on image shape | 0.001 | 0.7 | 100 |
| detr | Transformer; facebook/detr-resnet-50 | Processor default: short side 800, long side ≤1333 | 0.001 | None | 100 queries |

Faster R-CNN uses region proposals followed by box classification/refinement; YOLO11s
uses a one-stage prediction pipeline. DETR treats detection as set prediction with
learned queries and bipartite matching during pretraining, avoiding NMS. See the
[torchvision model documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html),
[YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/) and
[DETR paper](https://arxiv.org/abs/2005.12872).

These are comparisons of deployed configurations. Input resolutions, parameter counts,
training recipes and backend implementations differ, so the results do not isolate
architecture alone. Low confidence thresholds retain detections for ranking in AP;
family-specific NMS defaults remain explicit. No fine-tuning or accelerator export was used.

## Datasets and taxonomy

| Evaluation | Images | Split | Classes |
| --- | --- | --- | --- |
| COCO-80 | 5,000 | COCO val2017 | Native 80 classes |
| COCO-2 | 5,000 | Same COCO val2017 images | vehicle / person |
| KITTI-2 | 7,481 | All labeled KITTI object-detection training images | vehicle / person |

KITTI's labeled training split is used exclusively for evaluation: no weights are
trained or selected on it, and the unlabeled test set is unused. Every model sees
the same sorted image IDs per dataset. Images without target objects remain included.

| Shared class | COCO ground truth and detector names | KITTI ground truth |
| --- | --- | --- |
| vehicle | car, truck, bus, train | Car, Van, Truck, Tram |
| person | person | Pedestrian, Person_sitting |

The same mapping is applied to ground truth and predictions. Other COCO classes and
their predictions are excluded from the two-class task. Native COCO-80 scores are
reported separately and are never subtracted from two-class KITTI AP.

KITTI has 33,261 mapped vehicles and 4,709 mapped persons. Cyclist (1,627), Misc (973),
and DontCare (11,295) boxes are represented as COCO crowd-ignore regions for both
classes. At each COCO overlap threshold, matching prioritizes real target objects;
remaining detections that match an ignore region are excluded using
intersection/detection-area. Regions are not deleted and then treated as background.
Boxes are clipped to image bounds and converted from xyxy to xywh without a +1 pixel
adjustment. All target objects are retained without KITTI difficulty filtering.

This is a custom **COCO-style KITTI evaluation**, not official KITTI AP_R40 or its
easy/moderate/hard scores. In particular, duplicating excluded-object regions for both
classes can suppress predictions overlapping those regions; this is an approximation
whose effect was not separately measured. COCO person/bicycle boxes do not directly
represent KITTI's combined Cyclist object. The shared taxonomy improves comparability
without making the two annotation schemes identical. See the
[KITTI evaluation specification](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)
and the [COCO evaluator implementation](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).

## Accuracy results

All AP/AR values are fractions in [0,1]. AP averages IoU thresholds .50:.05:.95;
AP50 and AP75 fix the overlap threshold. AR1/10/100 cap detections per category per
image. COCO small/medium/large use annotation-area thresholds 32² and 96² pixels.
COCO retains its supplied areas (normally segmentation area); KITTI uses bounding-box
area, which is another limitation for scale comparisons.

{accuracy_text}

## Recall results

{recall_text}

## Analysis 1: cross-dataset generalization

![Shared-class AP on COCO and KITTI](results/summary/figures/domain_accuracy.png)

{gap_table}

Positive differences mean AP is lower on KITTI. These are absolute percentage-point
differences, not relative percentage reductions. YOLO11s has the largest aggregate
drop in these runs; DETR has the smallest drop, but also starts from a lower COCO AP
and has the lowest KITTI AP. A smaller gap alone does not establish the best detector.

### Which classes degrade most?

{class_table}

![Class-specific AP drop](results/summary/figures/class_gap.png)

Person AP drops by 26–32 percentage points, versus about 3–6.5 points for vehicle AP.
This pattern is consistent across all three models. Smaller or occluded pedestrians,
viewpoint changes, and differences in annotation conventions are plausible explanations,
but these experiments do not separate their effects. No qualitative examples or
occlusion-stratified measurements were collected to establish a cause.

### Which scales degrade most?

{scale_table}

Medium and large objects show larger absolute AP reductions than small objects for
all three models. Small-object AP is nevertheless the lowest scale-specific AP within
KITTI for every model. Thus “small objects are hardest” and “small objects suffer the
largest cross-dataset drop” are different claims; these results support the first.
Area definitions and object composition differ between datasets, so scale gaps should
be interpreted as descriptive comparisons rather than controlled effects.

KITTI AP75 also drops more than AP50 for every model, suggesting stricter localization
is more sensitive in this transfer setting. Annotation geometry can contribute, so
this observation is not proof of a particular architectural failure.

## Analysis 2: speed, accuracy, and GPU memory

![Speed versus accuracy; class count is stated separately for each panel](results/summary/figures/speed_accuracy.png)

{timing_table}

Latency includes image preprocessing, device transfers, model inference, postprocessing,
and conversion to CPU Python outputs through predict(). It excludes model loading,
disk reads, image decoding, and warmup. Each model runs in a separate process, with
batch size 1, 20 warmup predictions, 100 measured predictions over the first 100 sorted
images of its dataset, and CUDA synchronization before and after each timed call.
FPS is 1000 / mean latency; it is serial predict() throughput, not complete disk-to-result
pipeline throughput. p50 is median latency and p95 the 95th percentile.

Peak allocated memory counts live PyTorch tensors, including resident weights and
inference intermediates. Peak reserved also includes allocator caching. Peak counters
are reset after warmup. These figures exclude allocations outside PyTorch and are not
whole-device nvidia-smi memory or activation-only memory.

YOLO11s is the fastest and uses the least allocated GPU memory on both datasets.
Faster R-CNN yields higher overall accuracy at greater latency and memory cost.
On native COCO, both Faster R-CNN and YOLO11s occupy the observed accuracy/latency
frontier; DETR has lower AP and higher latency than YOLO11s in this configuration.
On KITTI, Faster R-CNN also has the highest small/medium AP, while DETR's large-object
AP (0.5041) is slightly above Faster R-CNN's (0.4992); no uncertainty estimate was made
for that small difference.

Every model runs faster on the KITTI timing sample than on the COCO sample. Image
aspect ratios, internal resized tensor shapes, scene content and postprocessing work
can differ; these timings are not evidence that dataset identity itself causes speedup.
YOLO11s's COCO p95 is 38.32 ms despite a mean of 11.79 ms, indicating tail variability
in this single 100-prediction run. The raw samples do not identify its cause.

## Reproducibility

Experiments ran on PSC Bridges-2 with one {env['gpu']} GPU (32 GiB), NVIDIA driver
580.178.04, four allocated CPU cores, PyTorch CPU thread count {env['torch_num_threads']},
and native backend execution. COCO-80/timing job {args.native_dir.name} ran on v003;
shared-taxonomy/KITTI job {args.domains_dir.name} ran on v020. The nodes share the GPU
model but need not share host load. GPU nodes were in the GPU-shared partition.

| Component | Recorded version |
| --- | --- |
| Python | {env['python']} |
| torch | {env['torch']} |
| torchvision | {env['torchvision']} |
| CUDA build | {env['cuda']} |
| cuDNN reported by PyTorch | {env['cudnn']} |
| ultralytics | {env['ultralytics']} |
| transformers | {env['transformers']} |
| NumPy | {env['numpy']} |
| Pillow | {env['pillow']} |
| pycocotools | 2.0.11 |

The numeric cuDNN API version is preserved exactly as recorded. Complete dependency
snapshots are in [native-run packages]({args.native_dir}/packages.txt) and
[domain-run packages]({args.domains_dir}/packages.txt). requirements.txt lists broad
dependencies; those snapshots, rather than today's unpinned package resolution,
describe the actual experiment environments. The model identifiers name pretrained
checkpoints; immutable downloaded-weight hashes/HF revisions were not recorded.

### Run the experiments

From the cluster repository root, using an environment with these dependencies:

```bash
# Install the V100-compatible PyTorch build first.
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install --no-cache-dir -r requirements.txt

# Expected input paths:
# data/coco/val2017/*.jpg
# data/coco/annotations/instances_val2017.json
# data/kitti/training/image_2/*.png
# data/kitti/training/label_2/*.txt

sbatch scripts/run_coco.slurm
sbatch scripts/run_domains.slurm
```

The batch scripts use the project-specific Python path and PSC account; adjust those
for another system. Each submission creates its own job-ID result directory. Dataset
sources: [COCO downloads](https://cocodataset.org/#download) and
[KITTI object detection](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d).
Use KITTI's left-color images and labels, and extract the labeled training portion.

All **17 exact saved evaluation/benchmark/sanity commands**, plus KITTI conversion,
are in [commands.md](results/summary/commands.md). Individual JSONs also retain the
working directory, CLI arguments, selected image IDs, model settings and environment.
The original cluster paths differ from the local backup paths linked below.

### Artifacts and regeneration

- [Native COCO JSONs]({args.native_dir}/) and [shared COCO/KITTI JSONs]({args.domains_dir}/).
- [Accuracy CSV](results/summary/accuracy.csv): all 12 COCO statistics for every evaluation.
- [Per-class AP CSV](results/summary/per_class_ap.csv): all 80 native classes and shared classes.
- [Readable 80-class appendix](results/summary/coco_per_class.md).
- [Timing/memory CSV](results/summary/benchmarks.csv) and [generalization CSV](results/summary/generalization.csv).
- [Source-file SHA-256 manifest](results/summary/source_sha256.json) identifies the copied inputs.
- PNG figures appear above; PDF versions are in results/summary/figures/ for export.
- Logs: [COCO](results/coco_{args.native_dir.name}.log), [shared domains](results/domains_{args.domains_dir.name}.log).

To rebuild this report locally from the backed-up JSON files, without a GPU or datasets:

```bash
python scripts/make_report.py --native-dir {args.native_dir} --domains-dir {args.domains_dir}
```

Build the PDF and supporting ZIP with `python scripts/build_submission.py`.

The generator validates image counts/IDs, category counts, required metadata, sanity
scores, and timing summaries against the saved raw samples before writing the tables.

## Validation and limits

{table(['Ground-truth sanity test', 'Images', 'Aggregate AP', 'Both per-class APs'], sanity_rows)}

Feeding non-crowd ground truth back as predictions yields AP 1.0000 on both full
shared-taxonomy datasets. This checks evaluator plumbing, not model accuracy or the
scientific adequacy of the taxonomy. KITTI medium-area sanity AP rounds to 0.9999;
aggregate and both per-class APs round to 1.0000. Low AR1/AR10 in perfect-prediction
tests can occur because those metrics cap the number of detections.

Synthetic tests exercise mapping on predictions and ground truth, ignore-region
matching, negative images, zero-prediction scoring, and malformed input rejection:

```bash
python -m unittest discover -s scripts -p 'test_*.py'
```

Accuracy uses complete datasets, but timing covers one 100-image sample per dataset
without repeated independent runs, confidence intervals or power/clock controls.
COCO-to-KITTI changes class/object distributions and annotation practices as well as
visual domain. Label merging and KITTI's custom ignore policy affect interpretation.
No fine-tuning, open-vocabulary comparison or causal ablation is claimed. Training-set
provenance beyond the named COCO-pretrained checkpoints was not audited for overlap.

The report covers three configurations/families, two datasets, accuracy/recall and
per-class/scale metrics, latency/FPS, GPU memory, reproducibility details and the two
required technical analyses. Datasets, downloaded weights and smoke outputs must not
be committed. Keep both datasets and archives until the homework has been reviewed.
"""
    Path("README.md").write_text(readme)
    print(f"Validated {len(evaluations)} evaluations, {len(benchmarks)} benchmarks, and 2 sanity runs.")
    print("Wrote README.md, four CSV tables, exact commands, class appendix, source hashes, and three PNG/PDF figures.")


if __name__ == "__main__":
    main()
