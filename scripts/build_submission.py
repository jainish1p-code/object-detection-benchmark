"""Build the nine-page technical report and a small code/results submission ZIP.

Uses the existing matplotlib dependency; no LaTeX installation is required.
Run from the repository root: python scripts/build_submission.py
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "detection-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "submission"
PREVIEW = Path(tempfile.gettempdir()) / "object-detection-report-preview"
WIDTH, HEIGHT = 612, 792
MARGIN = 43
BODY_WIDTH = WIDTH - 2 * MARGIN
NAMES = {"frcnn": "Faster R-CNN", "yolo11s": "YOLO11s", "detr": "DETR"}
MODELS = tuple(NAMES)


class Page:
    """Measured text layout with overflow checks and a preview of every PDF page."""
    def __init__(self, number, title, pdf, markdown):
        self.number, self.pdf, self.markdown = number, pdf, markdown
        self.fig = plt.figure(figsize=(8.5, 11), dpi=120)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, WIDTH), ylim=(0, HEIGHT))
        self.ax.axis("off")
        self.fig.canvas.draw()
        self.renderer = self.fig.canvas.get_renderer()
        self.y = 716
        self.ax.text(MARGIN, 756, "OBJECT DETECTION / TRACK A", fontsize=8,
                     color="#365574", weight="bold")
        self.ax.text(MARGIN, 733, title, fontsize=16, weight="bold", va="top", color="#142c46")
        self.ax.plot([MARGIN, WIDTH - MARGIN], [720, 720], color="#d6dee6", lw=0.8)
        self.y = 699
        markdown.extend([f"# Page {number}: {title}", ""])

    def wrap(self, text, size, family="DejaVu Sans", weight="normal"):
        font = FontProperties(family=family, size=size, weight=weight)
        def width(value):
            return self.renderer.get_text_width_height_descent(value, font, False)[0] * 72 / self.fig.dpi
        lines, current = [], ""
        for word in text.split():
            if current and width(current + " " + word) > BODY_WIDTH:
                lines.append(current)
                current = ""
            # Long URLs must fit without escaping the page boundary.
            while width(word) > BODY_WIDTH:
                length = len(word)
                while width(word[:length]) > BODY_WIDTH:
                    length -= 1
                lines.append(word[:length])
                word = word[length:]
            current = (current + " " + word).strip()
        if current:
            lines.append(current)
        return lines

    def text(self, text, size=9.5, gap=8, family="DejaVu Sans", weight="normal", url=None):
        leading = size * 1.38
        for line in self.wrap(text, size, family, weight):
            self.ax.text(MARGIN, self.y, line, fontsize=size, va="top", family=family,
                         weight=weight, color="#17212b", url=url)
            self.y -= leading
        self.y -= gap
        self.markdown.extend([text, ""])

    def heading(self, text):
        self.y -= 2
        self.text(text, size=11.2, gap=5, weight="bold")

    def table(self, headers, rows, widths=None, size=8.1, row_height=21):
        widths = widths or [1 / len(headers)] * len(headers)
        if abs(sum(widths) - 1) > 1e-6:
            raise ValueError("table widths must sum to 1")
        self.markdown.extend(["| " + " | ".join(headers) + " |",
                              "| " + " | ".join(["---"] * len(headers)) + " |"])
        for index, row in enumerate([headers, *rows]):
            if index == 0:
                self.ax.add_patch(plt.Rectangle((MARGIN, self.y - row_height), BODY_WIDTH,
                                                row_height, color="#eaf0f6", zorder=-1))
            left = MARGIN
            for item, width in zip(row, widths):
                # Table cells are deliberately short; verify actual text width.
                font = FontProperties(size=size, weight="bold" if index == 0 else "normal")
                length = self.renderer.get_text_width_height_descent(str(item), font, False)[0] * 72 / self.fig.dpi
                if length > BODY_WIDTH * width - 7:
                    raise ValueError(f"table text too wide on page {self.number}: {item}")
                self.ax.text(left + 3, self.y - 6, str(item), fontsize=size, va="top",
                             weight="bold" if index == 0 else "normal")
                left += BODY_WIDTH * width
            self.y -= row_height
            self.ax.plot([MARGIN, WIDTH - MARGIN], [self.y, self.y], color="#d6dee6", lw=0.4)
            if index:
                self.markdown.append("| " + " | ".join(map(str, row)) + " |")
        self.y -= 11
        self.markdown.append("")

    def figure(self, filename, height, caption):
        image = plt.imread(ROOT / filename)
        width = min(BODY_WIDTH, height * image.shape[1] / image.shape[0])
        left = (WIDTH - width) / 2
        axes = self.fig.add_axes([left / WIDTH, (self.y - height) / HEIGHT,
                                 width / WIDTH, height / HEIGHT])
        axes.imshow(image)
        axes.axis("off")
        self.y -= height + 5
        self.text(caption, size=8.3, gap=8)
        self.markdown.extend([f"![{caption}](../{filename})", ""])

    def finish(self):
        if self.y < 44:
            raise ValueError(f"page {self.number} overflows: bottom={self.y:.1f}")
        self.ax.text(MARGIN, 27, "COCO → KITTI | measured results and reproducible experiments",
                     fontsize=7.3, color="#52616d")
        self.ax.text(WIDTH - MARGIN, 27, f"{self.number} / 9", fontsize=7.3,
                     color="#52616d", ha="right")
        self.pdf.savefig(self.fig)
        self.fig.savefig(PREVIEW / f"page_{self.number:02d}.png", dpi=120)
        plt.close(self.fig)


def main():
    os.chdir(ROOT)
    # Validate the input records and regenerate all tables before typesetting.
    subprocess.run([sys.executable, "scripts/make_report.py"], check=True)
    OUTPUT.mkdir(exist_ok=True)
    PREVIEW.mkdir(exist_ok=True)
    refs = json.loads(Path("report/references.json").read_text())
    if len(refs) < 8 or sum(r["kind"] == "research" for r in refs) < 4 or sum(r["kind"] == "official" for r in refs) < 2:
        raise ValueError("reference requirements are not met")
    def read(path):
        return json.loads(Path(path).read_text())
    native = {m: read(f"results/50149631/{m}_eval.json") for m in MODELS}
    domain = {(d, m): read(f"results/50151110/{d}_{m}_eval.json")
              for d in ("coco", "kitti") for m in MODELS}
    bench = {(d, m): read(f"results/{'50149631/' if d == 'coco' else '50151110/kitti_'}{m}_benchmark.json")
             for d in ("coco", "kitti") for m in MODELS}
    md = ["# 2D Object Detection: COCO to KITTI", "", "Track A technical report. November 2, 2026.", ""]
    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    pdf_path = OUTPUT / "technical_report.pdf"
    with PdfPages(pdf_path, metadata={"Title": "2D Object Detection: COCO to KITTI",
                                     "Subject": "Track A experimental model evaluation",
                                     "Author": "", "Keywords": "COCO, KITTI, Faster R-CNN, YOLO11, DETR"}) as pdf:
        p = Page(1, "2D Object Detection: COCO to KITTI", pdf, md)
        p.text("Technical report · Track A · November 2, 2026", size=9, gap=13)
        p.heading("Abstract")
        p.text("This experiment compares three COCO-pretrained detector configurations: Faster R-CNN ResNet-50 FPN v2, YOLO11s, and DETR ResNet-50. Accuracy is evaluated on all 5,000 COCO val2017 images and all 7,481 labeled KITTI object-detection training images, without retraining. A shared vehicle/person taxonomy permits a controlled label-space comparison alongside native 80-class COCO evaluation. All measurements were produced by actual Slurm jobs on a Tesla V100 32 GB GPU, with saved JSON metrics and logs. Faster R-CNN has the highest aggregate AP; YOLO11s has the lowest latency and allocated GPU memory. Person AP degrades much more than vehicle AP on KITTI. The report analyzes transfer and efficiency while documenting annotation, ignore-region, resolution, and sampling limitations.")
        p.heading("1. Questions and experimental design")
        p.text("The two primary questions are: (1) How well do COCO-pretrained detectors transfer to autonomous-driving imagery when class definitions are aligned? (2) How much accuracy is traded for lower latency and memory? Each model is evaluated with the same image set, confidence threshold, and evaluator within each task. Family-specific preprocessing and NMS defaults are recorded rather than silently equated. The course tutorial [11] motivates separating detector adapters, dataset conversion, taxonomy, and evaluation.")
        p.heading("2. Detector families and their technical differences")
        p.text("Faster R-CNN shares image features between a region proposal network and a second-stage classifier/box regressor [1]. The evaluated torchvision v2 configuration uses a ResNet-50 feature pyramid [7]. FPN combines top-down features and lateral connections across scales [2]; the observed small-object results can be discussed in this context, but the experiment does not isolate FPN's causal contribution.")
        p.text("YOLO11s is a recent one-stage detector from Ultralytics [8]. The original YOLO paper established direct full-image box/class prediction [3]; it is cited as architectural background, not as a description of every YOLO11 layer or training loss. The actual evaluated 2024-era YOLO11 configuration is specified by its official implementation and checkpoint.")
        p.text("DETR predicts a set of boxes using learned object queries and a transformer encoder-decoder [4]. Bipartite matching is part of its pretraining objective; the inference path does not apply NMS. The Hugging Face adapter uses the pretrained ResNet-50 checkpoint and processor [9]. No model is fine-tuned here, and model capacity, training recipes, resolutions, and software paths differ. Findings therefore compare these three configurations, not detector families in isolation.")
        p.finish()

        p = Page(2, "Experimental setup, datasets, and metrics", pdf, md)
        p.heading("3. Models, hardware, and environment")
        p.table(["Configuration", "Checkpoint", "Resize / NMS"], [
            ["Faster R-CNN", "torchvision COCO_V1", "800 / ≤1333; IoU 0.5"],
            ["YOLO11s", "yolo11s.pt", "imgsz=640; IoU 0.7"],
            ["DETR ResNet-50", "facebook/detr-resnet-50", "800 / ≤1333; no NMS"]], widths=[.25, .39, .36], size=8.2)
        p.text("Faster R-CNN's exact model is fasterrcnn_resnet50_fpn_v2. The 800/≤1333 rule sets the target short side and limits the long side. YOLO uses aspect-preserving letterboxing with shape-dependent rectangular padding. Confidence is 0.001 for every configuration; adapters return at most 100 detections (100 queries for DETR). Defaults are recorded in the original JSONs. Native inference was used without TensorRT, explicit FP16, or fine-tuning.")
        p.text("PSC Bridges-2: Tesla V100-SXM2-32GB, NVIDIA driver 580.178.04, four allocated CPU cores, and four PyTorch CPU threads. Software: Python 3.11.18, torch 2.7.1+cu118, torchvision 0.22.1+cu118, CUDA build 11.8, Ultralytics 8.4.198, Transformers 5.22.1, pycocotools 2.0.11, NumPy 2.4.6. Exact package snapshots accompany both jobs. The CUDA 11.8 installation follows the official PyTorch version matrix [12].")
        p.heading("4. Dataset selection and label alignment")
        p.text("COCO [5] is evaluated on all 5,000 val2017 images. KITTI [6] uses all 7,481 labeled object-detection training images exclusively for evaluation. No weights or thresholds were fitted on KITTI; the unlabeled test split is unused. Every model sees identical sorted IDs within a dataset, including images with no mapped target objects. All image/label pairs were checked before evaluation.")
        p.table(["Shared label", "COCO GT and predictions", "KITTI GT"], [
            ["vehicle", "car, truck, bus, train", "Car, Van, Truck, Tram"],
            ["person", "person", "Pedestrian, Person_sitting"]], widths=[.18, .38, .44], size=8.1)
        p.text("Both ground truth and predictions are mapped. KITTI has 33,261 mapped vehicles and 4,709 persons. Cyclist (1,627), Misc (973), and DontCare (11,295) boxes become COCO crowd-ignore regions for both target classes. Real ground truth has matching priority; unmatched detections can be ignored by intersection/detection-area at each overlap threshold [10]. This custom policy is an approximation, not official KITTI AP_R40. All target difficulties are retained. Native COCO-80 evaluation is preserved separately.")
        p.heading("5. Metric definitions")
        p.text("AP averages interpolated precision over IoU=.50:.05:.95 and recall; AP50/AP75 fix IoU. AR1/10/100 average recall with detection caps per category/image [10]. Small, medium, and large use annotation areas split at 32² and 96² pixels. COCO retains supplied areas, normally segmentation areas; KITTI uses clipped bounding-box area. This difference limits scale comparisons. AP values are fractions in [0,1]; gaps are absolute percentage points.")
        p.finish()

        p = Page(3, "Quantitative accuracy and recall", pdf, md)
        p.text("Table 1. Full-dataset AP. C80 = native COCO (80 classes); C2 = COCO vehicle/person; K2 = KITTI vehicle/person. All three use the same COCO-style evaluator. Native C80 scores are not used to compute cross-domain gaps.", size=9)
        rows = []
        recall_rows = []
        for label, records in [("C80", native), ("C2", {m:domain['coco', m] for m in MODELS}), ("K2", {m:domain['kitti', m] for m in MODELS})]:
            for m in MODELS:
                r = records[m]["stats"]
                rows.append([label, NAMES[m], *(f"{r[k]:.4f}" for k in ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"))])
                recall_rows.append([label, NAMES[m], *(f"{r[k]:.4f}" for k in ("AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"))])
        widths = [.07, .21, .12, .12, .12, .12, .12, .12]
        p.table(["Set", "Model", "AP", "AP50", "AP75", "AP-S", "AP-M", "AP-L"], rows, widths=widths, size=7.9, row_height=19)
        p.text("Table 2. Recall over IoU=.50:.95. AR-S/M/L use maxDets=100. The lower shared-task AR1 is consistent with merging multiple vehicle categories under a single per-category detection cap; it should not be compared directly with native AR1 as if the tasks were identical.", size=9)
        p.table(["Set", "Model", "AR1", "AR10", "AR100", "AR-S", "AR-M", "AR-L"], recall_rows, widths=widths, size=7.9, row_height=19)
        p.text("Faster R-CNN has the highest aggregate AP in all three tasks. YOLO11s exceeds DETR's aggregate AP, but this does not hold for every class, scale, or overlap threshold. DETR has slightly higher KITTI AP50 than YOLO11s and the highest KITTI large-object AP. Small numerical differences have not been tested for statistical significance.")
        p.text("All 80 native COCO per-class AP values are included in results/summary/coco_per_class.md and per_class_ap.csv. Shared-class AP is reported on page 4. Raw JSONs preserve all 12 aggregate metrics, selected image IDs, checkpoint settings, and exact commands; the supporting archive contains them rather than relying on copied tables alone.", size=9)
        p.finish()

        p = Page(4, "Analysis I: cross-dataset generalization", pdf, md)
        p.figure("results/summary/figures/domain_accuracy.png", 207,
                 "Figure 1. Shared-class AP on full COCO and KITTI; both sides use vehicle/person.")
        gap_rows = []
        class_rows = []
        for m in MODELS:
            c, k = domain['coco',m], domain['kitti',m]
            gap_rows.append([NAMES[m], f"{c['stats']['AP']:.4f}", f"{k['stats']['AP']:.4f}",
                             f"{100*(c['stats']['AP']-k['stats']['AP']):.2f}"])
            class_rows.append([NAMES[m], *(f"{r['per_class_ap'][name]:.4f}" for name in ('vehicle','person') for r in (c,k))])
        p.table(["Model", "COCO-2 AP", "KITTI-2 AP", "Drop (pp)"], gap_rows, widths=[.34,.22,.22,.22])
        p.table(["Model", "COCO veh.", "KITTI veh.", "COCO person", "KITTI person"], class_rows,
                widths=[.28,.17,.17,.19,.19], size=8)
        p.text("All three models transfer less effectively to KITTI under this protocol. YOLO11s has the largest aggregate drop (19.16 points); DETR has the smallest (14.46), but its KITTI AP is still the lowest. Transfer gap alone is not a sufficient selection criterion: a weaker source-domain baseline can also have a smaller absolute gap.")
        p.text("Person AP falls by 29.18 points for Faster R-CNN, 31.85 for YOLO11s, and 26.13 for DETR. Vehicle AP falls by only 6.20, 6.48, and 2.80 points, respectively. The consistent class-specific pattern is stronger evidence than a single pooled AP change. It identifies pedestrian-related detection as the most important transfer weakness in this experiment.")
        p.text("The label-space control does not isolate visual domain shift completely. COCO's person annotations, KITTI's pedestrian/sitting-person definitions, merged vehicle categories, ignored cyclists, object distributions, and annotation geometry differ. Viewpoint, occlusion, and smaller distant pedestrians are plausible mechanisms, not measured causal explanations. No retraining was performed to test whether adaptation removes these deficits.", size=9.2)
        p.finish()

        p = Page(5, "Failure analysis: scales and localization", pdf, md)
        p.heading("6. Quantitative failure symptoms")
        p.text("Failure analysis uses measured class, scale, and overlap-threshold behavior. Per-image predictions were not retained, so false-positive/negative counts and qualitative examples cannot be reconstructed from these artifacts. Aggregate AP and AR can reveal weaknesses but cannot uniquely distinguish missed objects, misclassification, duplicate boxes, and localization errors.")
        scale_rows = []
        localization_rows = []
        for m in MODELS:
            c, k = domain['coco',m]['stats'], domain['kitti',m]['stats']
            scale_rows.append([NAMES[m], *(f"{100*(c[key]-k[key]):.2f}" for key in ('AP_small','AP_medium','AP_large'))])
            localization_rows.append([NAMES[m], f"{k['AP50']:.4f}", f"{k['AP75']:.4f}",
                                      f"{100*(k['AP50']-k['AP75']):.2f}", f"{k['AR100']:.4f}"])
        p.table(["Model", "Small drop (pp)", "Medium drop (pp)", "Large drop (pp)"], scale_rows,
                widths=[.28,.24,.25,.23], size=8)
        p.text("Medium/large objects have larger absolute cross-domain AP losses than small objects for every model. Small-object KITTI AP is nevertheless lowest within each model: 0.2720, 0.1979, and 0.1672. These are different observations: difficult small objects do not necessarily have the greatest absolute transfer loss. COCO segmentation-area bins and KITTI box-area bins are imperfectly comparable, and scale-specific class composition can change.")
        p.table(["Model", "KITTI AP50", "KITTI AP75", "Gap (pp)", "AR100"], localization_rows,
                widths=[.30,.18,.18,.18,.16], size=8)
        p.text("The large AP50-to-AP75 gaps on KITTI show that accepting roughly overlapping boxes is much easier than requiring tighter overlap. AP75 declines more than AP50 from shared COCO to KITTI for every model. This is consistent with a localization-sensitive weakness, but changes in box conventions and ranked prediction quality can also contribute. KITTI AR100 remains below 0.53, showing incomplete recall averaged across overlap thresholds; it is not a literal percentage of objects missed at one threshold.")
        p.heading("7. Pipeline checks and known evaluator limitations")
        p.text("Full ground-truth-as-prediction sanity runs give aggregate AP and both class APs of 1.0000 on COCO and KITTI. KITTI medium-area AP rounds to 0.9999, which is disclosed rather than presented as exactly perfect at every submetric. Four synthetic tests cover shared-label mapping, negative images, crowd-ignore behavior, zero predictions, and malformed inputs. These validate mechanics, not the suitability of the experimental taxonomy.")
        p.text("Duplicating Cyclist/Misc/DontCare boxes as ignore regions for both classes can suppress detections that overlap those regions, including real target objects. Its effect was not ablated. Official KITTI difficulty filtering and AP_R40 are not used. A stronger follow-up would retain per-image predictions, inspect false positives and missed pedestrians, separate truncation/occlusion strata, and repeat evaluation under alternate ignore policies.", size=9.2)
        p.finish()

        p = Page(6, "Analysis II: speed, accuracy, and memory", pdf, md)
        p.figure("results/summary/figures/speed_accuracy.png", 180,
                 "Figure 2. Accuracy/latency tradeoff; COCO uses 80 classes and KITTI uses two, in separate panels.")
        rows = []
        for d in ('coco','kitti'):
            for m in MODELS:
                b=bench[d,m]
                rows.append([d.upper(), NAMES[m], *(f"{b['latency_ms'][key]:.2f}" for key in ('mean','p50','p95')),
                             f"{b['fps']:.2f}", f"{b['gpu_memory']['peak_allocated_mib']:.1f}",
                             f"{b['gpu_memory']['peak_reserved_mib']:.0f}"])
        p.table(["Set", "Model", "Mean ms", "p50 ms", "p95 ms", "FPS", "Alloc MiB", "Res MiB"], rows,
                widths=[.09,.21,.12,.11,.11,.10,.14,.12], size=7.4, row_height=19)
        p.text("Each independent model process performs 20 warmup predictions, then 100 timed predictions on the first 100 sorted images, batch size 1. CUDA synchronization brackets each predict() call. Timing includes preprocessing, GPU transfers, inference, postprocessing, and CPU Python outputs; disk reads, decoding, loading, and warmup are excluded. FPS = 1000/mean-ms is serial predict() throughput, not a complete application pipeline rate.", size=9)
        p.text("Peak allocated memory includes resident weights and live inference tensors. Reserved memory includes the PyTorch allocator cache; both peaks are reset after warmup. Non-PyTorch allocations are excluded. These are not whole-device nvidia-smi values or activation-only measurements.", size=9)
        p.text("YOLO11s has 5.44× Faster R-CNN's COCO FPS at a 1.43-point native-COCO AP cost, and 6.02× its KITTI FPS at a 3.14-point KITTI AP cost. Its allocated GPU memory is also lowest. Faster R-CNN supplies higher aggregate accuracy; DETR has lower aggregate accuracy and higher latency than YOLO11s in these configurations. This is an observed configuration tradeoff, not proof that one family always dominates.", size=9)
        p.text("YOLO11s COCO p95 is 38.32 ms versus p50 9.97 ms, exposing tail variability hidden by average FPS. One timing pass cannot establish the cause. All models are faster on the KITTI timing sample; aspect ratios, resized shapes, scene content, and backend work can vary. No repeated-run intervals, GPU power/clock controls, or timing-sample representativeness analysis were performed.", size=9)
        p.finish()

        p = Page(7, "Reproduction and evidence of execution", pdf, md)
        p.heading("8. Actual runs and supporting artifacts")
        p.table(["Job", "Compute node", "Recorded scope / outcome"], [
            ["50149631", "v003 / V100", "Native COCO + benchmarks; exit 0:0"],
            ["50151110", "v020 / V100", "COCO/KITTI shared task; completion log"]],
            widths=[.17,.25,.58], size=8.1)
        p.text("Native job 50149631 completed in 00:15:25 according to the recorded Slurm status. Domain log 50151110 spans 18:13:07–18:53:44 EST on November 2 and ends with the script's completion marker. All 9 evaluation JSONs, 6 benchmark JSONs, 2 sanity JSONs, package snapshots, and both logs were copied from Ocean to the local project. Results are measured outputs, not numbers transcribed from model documentation.")
        p.heading("9. Code paths and commands")
        p.text("src/detectors.py implements the three adapters; taxonomy.py aligns label names; prepare_kitti.py validates and converts labels. eval_coco.py computes metrics, records image IDs/settings, and saves per-class AP. benchmark.py measures synchronized latency and memory. scripts/run_coco.slurm and run_domains.slurm request one GPU and create job-specific output directories. Every major saved JSON records its original exact command and working directory.")
        p.text("From the cluster repository root, submit the original configurations:", size=9, gap=4)
        p.text("sbatch scripts/run_coco.slurm", size=8, family="DejaVu Sans Mono", gap=2)
        p.text("sbatch scripts/run_domains.slurm", size=8, family="DejaVu Sans Mono", gap=8)
        p.text("An exact recorded KITTI evaluation command (display-wrapped; one shell command):", size=9, gap=4)
        p.text(domain['kitti','frcnn']['command'], size=7.7, family="DejaVu Sans Mono", gap=9)
        p.text("The original commands preserve account-specific Python and output paths. Re-running the batch scripts generates new job-ID directories. Installation uses torch 2.7.1 / torchvision 0.22.1 from the cu118 index [12] followed by requirements.txt; the two saved packages.txt files define the observed full environment more precisely than unpinned dependency resolution.", size=9)
        p.heading("10. How to inspect the submitted evidence")
        p.text("The companion experiment_code_and_results.zip contains executable source and Slurm configurations, raw metrics with image IDs, package snapshots, logs, exported CSVs, figures, references, and exact commands. Dataset images, downloaded weights, and results/smoke are excluded. The SHA-256 manifest identifies the raw report inputs; it is not an independent timestamp or proof of model training provenance.")
        p.text("Start with README.md and results/summary/commands.md. Native backups are in results/50149631/ and shared-domain backups in results/50151110/. accuracy.csv contains all AP/AR statistics; per_class_ap.csv and coco_per_class.md contain all 80 native classes plus shared-class results. benchmarks.csv preserves timing and allocated/reserved memory. No dataset images are needed to regenerate the report from these artifacts.")
        p.text("Regenerate tables/plots: python scripts/make_report.py. Regenerate this PDF and archive: python scripts/build_submission.py. Validate synthetic evaluator behavior: python -m unittest discover -s scripts -p 'test_*.py'. The report generator also checks counts/IDs, metadata, sanity results, and timing summaries against raw samples.", size=9)
        p.finish()

        p = Page(8, "Conclusions, limitations, and AI assistance", pdf, md)
        p.heading("11. Conclusions supported by the experiment")
        p.text("First, matching a shared taxonomy makes a cross-dataset experiment more interpretable than subtracting 80-class COCO AP from two-class KITTI AP. With this control, every model loses AP, and person AP declines much more than vehicle AP. The model ranking remains Faster R-CNN, YOLO11s, DETR by aggregate AP. A smaller transfer gap does not compensate for a lower target-domain score.")
        p.text("Second, accuracy and deployment cost favor different configurations. Faster R-CNN offers the strongest aggregate accuracy, while YOLO11s offers the best measured latency and memory. The practical choice depends on application constraints, including tail latency. DETR is neither the fastest nor the most accurate here; its conceptual simplification of prediction [4] does not imply a speed win for this checkpoint and software path.")
        p.text("Third, the failure analysis distinguishes class, scale, and localization symptoms. People transfer worse than vehicles; small KITTI objects are hardest within each model, but medium/large bins have larger absolute cross-domain AP losses. The AP75 decline motivates checking box alignment and localization quality rather than attributing every loss to class recognition. These are descriptive, testable findings rather than causal claims.")
        p.heading("12. Threats to validity and useful next experiments")
        p.text("The models differ in training recipes, parameter counts, input sizes, and backend implementations. Results compare complete configurations. COCO and KITTI differ in visual content, class/object distributions, annotation conventions, and area definitions. Merged labels and crowd-ignore handling can affect rankings. The experiment measures no sensitivity to alternate ignore policies, thresholds, label mappings, or input resolutions.")
        p.text("Latency comes from one 100-image pass per model/dataset rather than repeated trials. The first sorted images may not be representative. GPU-shared nodes and different compute hosts can introduce load variability. CUDA synchronization and warmup prevent basic timing errors but do not establish confidence intervals. Peak memory excludes non-PyTorch allocations and initialization peaks before the reset.")
        p.text("Checkpoint identifiers and framework versions are recorded, but immutable downloaded-weight hashes and a specific Hugging Face checkpoint revision were not. Pretraining provenance was not audited for overlap beyond the named COCO-pretrained weights. No training is performed here; using KITTI's labeled training split is an evaluation protocol choice, not a fine-tuning experiment or official leaderboard submission.")
        p.text("Prioritized follow-ups are: preserve per-image predictions for visual/error-stratified audits; repeat latency on randomized samples with clock/load metadata; ablate ignore-region treatment and area definitions; then evaluate fine-tuning or a newer detector under the same protocol. These were not run and are not included as measured results.")
        p.heading("13. AI tool usage and verification")
        p.text("Claude supplied initial handoff notes. ChatGPT/Codex assisted with experiment planning, environment debugging, benchmark/evaluation/conversion scripts, reference discovery, analysis, report generation, and packaging. The user ran the actual cluster commands and copied the experiment outputs. Report numbers are computed from those outputs; code tests and full-dataset sanity checks validate parts of the pipeline. Architectural explanations were checked against original papers and official sources cited here. The student remains responsible for reviewing the generated code, explanations, and final submission; AI assistance does not replace that review.", size=9.1)
        p.finish()

        p = Page(9, "References", pdf, md)
        p.text("Six original research papers [1–6], five official repository/framework sources [7–10,12], and the course tutorial [11]. Foundational papers provide background; current YOLO11 details are attributed to the official model documentation. Web documentation accessed November 2, 2026.", size=9, gap=10)
        for reference in refs:
            p.text(f"[{reference['id']}] {reference['citation']}", size=8.5, gap=1)
            p.text(reference['url'], size=7.3, gap=7, url=reference['url'])
        p.finish()

    (OUTPUT / "technical_report.md").write_text("\n".join(md))
    # Package only explicit source/result paths; never traverse data or weights.
    files = [Path("README.md"), Path("NOTES.md"), Path("AGENTS.md"), Path("requirements.txt"), Path(".gitignore")]
    files += sorted(Path("src").glob("*.py"))
    files += sorted(Path("scripts").glob("*.py")) + sorted(Path("scripts").glob("*.slurm"))
    files += sorted(Path("report").glob("*.json"))
    for folder in (Path("results/50149631"), Path("results/50151110"), Path("results/summary")):
        files += sorted(p for p in folder.rglob("*") if p.is_file())
    files += [Path("results/coco_50149631.log"), Path("results/domains_50151110.log")]
    with zipfile.ZipFile(OUTPUT / "experiment_code_and_results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            if not path.is_file():
                raise FileNotFoundError(path)
            if "data" in path.parts or "smoke" in path.parts or path.suffix in {".pt", ".pth", ".zip"}:
                raise ValueError(f"excluded large artifact selected: {path}")
            archive.write(path, path.as_posix())
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (pdf_path, OUTPUT / "experiment_code_and_results.zip")}
    (OUTPUT / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print("Built submission/technical_report.pdf (9 pages), Markdown source, and experiment_code_and_results.zip")
    print("Page previews:", PREVIEW)


if __name__ == "__main__":
    main()
