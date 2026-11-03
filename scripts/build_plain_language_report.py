"""Build an independent plain-language PDF from the saved experiment results.

Run from the repository root: python scripts/build_plain_language_report.py
The original technical PDF and supporting archive are not rebuilt or overwritten.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import build_submission as layout
from build_submission import MODELS, NAMES, Page, PdfPages, ROOT, plt


def main():
    os.chdir(ROOT)
    output = ROOT / "submission"
    figure_dir = output / "plain_language_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    layout.PREVIEW = Path(tempfile.gettempdir()) / "object-detection-plain-language-preview"
    layout.PREVIEW.mkdir(exist_ok=True)
    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans"})

    def read(path):
        return json.loads(Path(path).read_text())

    # Check the same raw inputs that support the technical report.
    for path, digest in read("results/summary/source_sha256.json").items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Experiment input changed: {path}")
    native = {m: read(f"results/50149631/{m}_eval.json") for m in MODELS}
    domain = {(d, m): read(f"results/50151110/{d}_{m}_eval.json")
              for d in ("coco", "kitti") for m in MODELS}
    timing = {(d, m): read(f"results/{'50149631/' if d == 'coco' else '50151110/kitti_'}{m}_benchmark.json")
              for d in ("coco", "kitti") for m in MODELS}
    references = read("report/references.json")

    x = list(range(len(MODELS)))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for offset, dataset, title, color in [(-.18, "coco", "Everyday photos", "#2563eb"),
                                         (.18, "kitti", "Driving photos", "#dc6803")]:
        bars = ax.bar([i + offset for i in x], [100 * domain[dataset,m]["stats"]["AP"] for m in MODELS],
                      width=.36, color=color, label=title)
        ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set(xticks=list(x), xticklabels=[NAMES[m] for m in MODELS], ylim=(0,72),
           ylabel="Detection score (AP × 100; higher is better)",
           title="The same models score lower on driving photos")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "everyday_vs_driving.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    for offset, category, title, color in [(-.18,"vehicle","Vehicles","#2563eb"),
                                          (.18,"person","People","#dc6803")]:
        values = [100 * (domain['coco',m]['per_class_ap'][category] -
                         domain['kitti',m]['per_class_ap'][category]) for m in MODELS]
        bars = ax.bar([i + offset for i in x], values, width=.36, color=color, label=title)
        ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set(xticks=list(x), xticklabels=[NAMES[m] for m in MODELS], ylim=(0,39),
           ylabel="Detection-score loss (points)", title="Recognizing people transfers less well")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "people_vs_vehicles.png", dpi=180)
    plt.close(fig)

    md = ["# When object-detection AI moves from everyday photos to driving scenes", "",
          "Plain-language report · November 2, 2026", ""]
    pdf_path = output / "plain_language_report.pdf"
    with PdfPages(pdf_path, metadata={"Title": "Object Detection: A Plain-Language Report",
                                      "Author": "", "Subject": "COCO-to-KITTI experiment explained for non-technical readers"}) as pdf:
        p = Page(1, "Can image-recognition AI handle a new setting?", pdf, md)
        p.text("A plain-language guide to the experiment · November 2, 2026", size=9.5, gap=13)
        p.heading("The question")
        p.text("An AI system may recognize objects well in ordinary photographs and still struggle when a camera looks out onto a road. This project tested that change of setting. It also asked a practical question: which system gives the best balance between recognition quality, processing time, and computer memory?")
        p.text("Three existing systems were tested: Faster R-CNN, YOLO11s, and DETR. They had already learned from a large collection of everyday photographs. They were used as supplied, without extra training on the driving photographs. The same saved experiment results support this report and the separate technical report.")
        p.heading("What the experiment found")
        p.table(["Finding", "What it means"], [
            ["Faster R-CNN scored highest", "Best overall recognition score in these tests"],
            ["YOLO11s was fastest", "Less time and measured GPU memory per image"],
            ["Every model lost score on driving photos", "Success in one setting did not transfer fully"],
            ["People had the larger score loss", "The change of setting affected classes differently"]],
            widths=[.48,.52], size=8.1, row_height=24)
        p.heading("How much evidence is behind this?")
        p.text("Recognition quality was measured on 5,000 everyday photographs and 7,481 driving photographs. These are complete selected dataset splits, rather than a handful of attractive examples. Processing speed and memory were measured on a smaller sample of 100 images from each collection. Actual computer-job logs, settings, and numerical result files were saved.")
        p.text("The main conclusion is a tradeoff. The model with the best recognition score was not the fastest. More importantly, none of the three retained its everyday-photo score when moved to the driving collection. The results help compare these model configurations; they do not establish a universal winner for every camera, scene, or application.")
        p.heading("How to read this report")
        p.text("Pages 2–3 explain the systems and the scoring in everyday terms. Pages 4–6 cover the findings and where the systems struggle. Pages 7–8 explain the checks, practical implications, and limits. Page 9 lists the research and official software sources. Technical terms are introduced where needed; the original report and code retain the detailed settings.")
        p.finish()

        p = Page(2, "What the systems do, and what we tested", pdf, md)
        p.heading("Object detection is more than naming a picture")
        p.text("A detector must say what an object is and draw a rectangle around where it appears. A driving image might contain several cars and pedestrians at once. A useful result needs the right label, the right number of objects, and boxes that fit them reasonably well. Calling the whole picture 'a road scene' would not solve this task.")
        p.heading("Three ways of approaching the task")
        p.text("Faster R-CNN works in two main stages: propose promising regions, then examine them to classify objects and refine their boxes [1,7]. Its feature pyramid combines information at several image scales [2]. The actual version tested is called fasterrcnn_resnet50_fpn_v2. Its name identifies a particular configuration, not every possible Faster R-CNN model.")
        p.text("YOLO11s uses a one-stage detection approach [8]. The original YOLO research introduced predicting boxes and object classes directly from a full image [3]. YOLO11s is a newer, specific implementation in that family, released in 2024; its official documentation, rather than the original YOLO paper, describes the version tested here. The 's' identifies its small model variant.")
        p.text("DETR uses a transformer to predict a collection of objects and their boxes [4,9]. It considers relationships and image context. Its design learns to produce a set of predictions without the usual separate step that removes overlapping duplicate detections. This experiment used the supplied ResNet-50 version, not every model based on DETR.")
        p.heading("Two photograph collections")
        p.table(["Collection", "What it represents", "Images checked"], [
            ["COCO", "Varied everyday scenes", "5,000"],
            ["KITTI", "Images from a driving setting", "7,481"]], widths=[.20,.57,.23], size=8.5)
        p.text("COCO is an established object-recognition dataset [5]. KITTI was created to test computer vision in autonomous-driving settings [6]. The models had learned from COCO before this project. Here, KITTI's labeled training images were used only for testing: no model was retrained or adjusted using them. The hidden-label KITTI test set was not used.")
        p.text("A move from one kind of image collection to another is often called a change of domain. It can involve different camera views, object sizes, backgrounds, and annotation rules. A lower score after that move shows weaker transfer under the chosen test, but does not by itself identify which difference caused it.")
        p.finish()

        p = Page(3, "What does a detection score mean?", pdf, md)
        p.heading("The errors the score needs to consider")
        p.text("A detector can miss a real object, report an object that is not there, name it incorrectly, or draw a poorly fitting box. It also gives a confidence score for each prediction. Testing must account for both the objects it finds and the false alarms it produces as predictions are ranked by confidence.")
        p.text("The main research measure is mean average precision, abbreviated mAP. It summarizes this ranked performance across the object categories and several requirements for box overlap. This report displays it on a 0–100 point scale: a saved mAP of 0.4694 becomes 46.94 points. Higher is better.")
        p.heading("A score is not a percentage of objects found")
        p.text("A score of 46.94 does NOT mean that the system correctly recognized exactly 46.94% of the objects. It combines several aspects of performance. Likewise, a confidence of 0.9 is the model's own score, not a guarantee that the prediction has a 90% chance of being correct. These distinctions matter when reading the tables.")
        p.heading("How closely must a box fit?")
        p.text("Box overlap compares the area shared by the predicted rectangle and the reference rectangle with the area covered by either one. For illustration, if they share 8,000 square pixels and together cover 12,000, the overlap score is about 0.67. That passes a 0.50 requirement but fails a stricter 0.75 requirement. This numerical example explains the rule; it is not a measured failure case.")
        p.text("AP50 uses the looser 0.50 overlap requirement; AP75 uses 0.75. The headline mAP averages requirements from 0.50 to 0.95. A detector that finds roughly the right region can therefore score better on AP50 than on AP75. Recall is a separate measure of how much of the labeled content can be recovered; false alarms are not summarized by recall alone [10].")
        p.heading("Two different comparisons in this project")
        rows = [[NAMES[m],f"{100*native[m]['stats']['AP']:.2f}"] for m in MODELS]
        p.table(["Model", "Everyday-photo score: all 80 categories"], rows, widths=[.34,.66], size=8.5)
        p.text("The table above is the broad COCO test. For the next page's transfer comparison, both datasets are instead scored using only two matched groups: vehicles and people. Their scores can differ simply because the task changed. We compare two-group scores with two-group scores, not the 80-category table with the driving scores.")
        p.finish()

        p = Page(4, "Finding 1: performance falls in the new setting", pdf, md)
        p.figure("submission/plain_language_figures/everyday_vs_driving.png", 205,
                 "Figure 1. Both collections use the same two target groups: vehicles and people.")
        rows = []
        for m in MODELS:
            c,k=domain['coco',m]['stats']['AP'],domain['kitti',m]['stats']['AP']
            rows.append([NAMES[m],f"{100*c:.2f}",f"{100*k:.2f}",f"{100*(c-k):.2f}"])
        p.table(["Model", "Everyday photos", "Driving photos", "Points lost"], rows,
                widths=[.31,.24,.24,.21], size=8.2)
        p.text("Every model scored lower on the driving collection. Faster R-CNN still had the highest overall score, followed by YOLO11s and DETR. The loss is meaningful because the object groups are aligned, all images in the selected splits are used, and every model follows the same evaluation rules within each dataset.")
        p.heading("A smaller drop does not automatically mean a better model")
        p.text("DETR lost the fewest points, but it began with a lower everyday-photo score and ended with the lowest driving-photo score. A model can have a smaller decline without being the strongest choice in the new setting. Both the starting point and the final score matter.")
        p.heading("What we controlled, and what still changed")
        p.text("Both reference labels and model predictions were grouped into vehicles and people. Cars, trucks, buses, and trains in COCO were aligned with KITTI's Car, Van, Truck, and Tram labels. KITTI's Pedestrian and Person_sitting labels were grouped as people. This removes a major mismatch in category names.")
        p.text("The collections still differ in scene content, object mix, camera viewpoint, and how people drew the reference boxes. Some labels and ignored regions have no perfect equivalent. The score decline is evidence of weaker transfer under this test; it is not a clean measurement of visual change alone.",size=9.2)
        p.finish()

        p = Page(5, "Where the models struggle", pdf, md)
        p.figure("submission/plain_language_figures/people_vs_vehicles.png", 192,
                 "Figure 2. People lose substantially more detection-score points than vehicles.")
        p.heading("Recognizing people was the clearest transfer weakness")
        p.text("Across the three models, the score for people fell by about 26–32 points. The vehicle score fell by about 3–6.5 points. This repeated pattern points to an important weakness that would be hidden by reporting only one overall score. It does not tell us the exact number of pedestrians missed.")
        p.heading("Small objects are difficult, but that is not the whole story")
        rows = [[NAMES[m], *(f"{100*domain['kitti',m]['stats'][key]:.1f}" for key in ('AP_small','AP_medium','AP_large'))] for m in MODELS]
        p.table(["Model", "Small objects", "Medium objects", "Large objects"],rows,
                widths=[.31,.23,.23,.23],size=8.3)
        p.text("These driving-photo scores show the weakest results for small objects. However, medium and large objects lost more points between the two collections. Being the hardest group in the final test is different from suffering the biggest decline. The dataset rules also measure object area differently, so these size comparisons are approximate.")
        p.heading("A roughly placed box is easier than a precise one")
        p.text("All models scored much better on driving photos when box overlap was judged loosely than when it was judged strictly. That suggests accurate placement deserves attention alongside recognizing the object category. The score alone cannot separate badly placed boxes from missed objects or other errors.")
        p.text("Different pedestrian viewpoints, partial obstruction, distant people, and labeling conventions are possible explanations for the weaknesses. They were not separately tested. Per-image predictions were not kept, so this report does not present individual failure images or claim to have counted each kind of error. This is a numerical failure analysis, with that limit made explicit.",size=9.2)
        p.finish()

        p = Page(6, "Finding 2: better scores come at a cost", pdf, md)
        p.heading("How long does one picture take?")
        p.text("A millisecond is one thousandth of a second. Lower processing time is better for speed. A graphics processor, or GPU, performs the model calculations; its memory holds model information and temporary working data. The measurements below come from a Tesla V100 with 32 GB of memory.")
        rows=[]
        for d,title in [('coco','Everyday'),('kitti','Driving')]:
            for m in MODELS:
                b=timing[d,m]
                rows.append([title,NAMES[m],f"{b['latency_ms']['mean']:.2f}",
                             f"{b['latency_ms']['p95']:.2f}",f"{b['fps']:.1f}"])
        p.table(["Photos", "Model", "Average ms", "95th-percentile ms", "Images / sec"],rows,
                widths=[.13,.23,.19,.28,.17],size=7.9,row_height=22)
        p.text("The 95th percentile describes the slower end of the sample: about 95% of measured predictions finished at or below that time. For YOLO11s on everyday photos, the average was 11.79 ms but this slower-end value was 38.32 ms. Average speed alone hides such variability. The sample cannot tell us why those slower predictions occurred.")
        p.heading("How much GPU memory was used?")
        p.table(["Model", "Everyday photos: MiB", "Driving photos: MiB"],
                [[NAMES[m],*(f"{timing[d,m]['gpu_memory']['peak_allocated_mib']:.1f}" for d in ('coco','kitti'))] for m in MODELS],
                widths=[.34,.33,.33],size=8.4)
        p.text("MiB is a unit of memory, close to a megabyte. These are peak amounts actively allocated by PyTorch, the software running the models. They include the loaded model and temporary tensors, but not every byte used by the GPU or the memory needed before warmup finished. Cached memory is recorded separately in the technical results.")
        p.heading("The practical tradeoff")
        p.text("On everyday photos, YOLO11s processed images about 5.4 times as fast as Faster R-CNN while giving up 1.43 points on the broad recognition score. On driving photos, it was about 6.0 times as fast with a 3.14-point score gap. It used the least measured GPU memory in both cases. Faster R-CNN offered the highest overall score; DETR sat between them in speed and memory.")
        p.text("Speed here means one image at a time through the prediction function. It includes image preparation and returning the boxes, but excludes reading image files from disk. It is not a measured complete video-processing rate. Differences in image size, model design, and software all contribute to the outcome.",size=9.2)
        p.finish()

        p = Page(7, "How trustworthy is the comparison?", pdf, md)
        p.heading("Evidence that the experiments actually ran")
        p.text("Two saved cluster jobs produced the measurements. Job 50149631 ran the broad everyday-photo tests and speed checks; Slurm reported successful completion. Job 50151110 ran the matched everyday/driving comparison and driving speed checks; its saved log ends with the completion message. The results include nine accuracy evaluations, six timing/memory measurements, and two full-data sanity checks.")
        p.heading("Checking the measuring tool before trusting the models")
        p.text("The reference answers were fed back into the evaluator as if they were predictions. This produced the maximum overall score on both collections. That checks that many parts of the scoring pipeline fit together. It does not mean the AI models achieved a perfect score or prove that every category-mapping choice is ideal. One size-specific sanity score rounded to 99.99 rather than 100; the technical report discloses it.")
        p.text("Four additional small tests checked label mapping, excluded regions, images without target objects, empty model outputs, and malformed inputs. All models used the same images and the same low confidence cutoff, while their usual image resizing and duplicate-removal rules were recorded. The code follows the course tutorial's separation of models, data, and evaluation [11].")
        p.heading("Accuracy and speed use different amounts of evidence")
        p.table(["Measurement", "Evidence used"], [
            ["Recognition quality", "All 5,000 everyday and 7,481 driving images"],
            ["Speed and memory", "100 timed predictions per model and collection"],
            ["Warmup", "20 untimed predictions before each speed test"],
            ["Repeatability", "Saved settings, commands, versions, IDs, and logs"]],
            widths=[.29,.71],size=8.1,row_height=23)
        p.text("The timing code waited for the GPU to finish before stopping each timer. Without this step, it might time the act of submitting work rather than the work itself. Warmup kept some first-use setup out of the measurements. However, the first 100 sorted images may not represent the full collection, and timing was not repeated across several independent samples.")
        p.heading("Important limits")
        p.text("These are three particular model versions, not equal-sized or identically trained systems. Their image resolutions differ. The GPU nodes were shared and the two jobs ran on different machines with the same GPU model. Small score differences and speed variations therefore deserve caution. The experiment has no repeated-run uncertainty estimates and does not establish performance on a different GPU, camera, or application.")
        p.finish()

        p = Page(8, "What to take away, and how to check it", pdf, md)
        p.heading("What the results support")
        p.text("If the priority is the highest overall recognition score among these tested configurations, Faster R-CNN leads. If processing speed and GPU memory are the main constraints, YOLO11s offers the strongest measured tradeoff. A single overall score is still insufficient: check the relevant object group, box-placement requirement, object size, and slower-end response times.")
        p.text("The strongest transfer finding is that people lost much more score than vehicles. That makes pedestrian-related detection a clear place for further investigation. It does not prove that a particular training method, model family, or camera change caused the loss. Testing on the intended setting remains necessary even when a system looks strong on a familiar benchmark.")
        p.heading("The driving score has a custom set of rules")
        p.text("Some KITTI labels, including cyclists and regions marked 'do not care,' do not fit the chosen two-group comparison cleanly. Their boxes were retained as ignored regions rather than counted as ordinary background. Predictions overlapping those regions can be left out of scoring for either target group. This prevents some unfair penalties, but can also suppress predictions near real target objects. The effect was not measured under an alternative rule.")
        p.text("For this reason, the reported driving scores should not be compared directly with KITTI's official leaderboard. Both photograph collections were scored using the same COCO-style evaluation framework, with the specific adaptation explained in the code and technical report [10]. The category grouping and reference-box conventions still leave differences between the two datasets.")
        p.heading("What to test next")
        p.text("Useful next steps would be to save predictions for individual images, inspect missed people and false alarms, test how ignored-region choices affect scores, and repeat speed checks on different image samples. Additional training on driving images could then be tested as a possible improvement. None of those follow-up experiments is claimed as completed here.")
        p.heading("Where the detail lives")
        p.text("The companion technical_report.pdf gives the full settings, metric tables, and limitations. experiment_code_and_results.zip includes the code, commands, raw results, package versions, and job logs. The code is also in github.com/jainish1p-code/object-detection-benchmark. To rerun the original jobs, the recorded commands are: sbatch scripts/run_coco.slurm and sbatch scripts/run_domains.slurm. Those require the datasets and the documented cluster environment; the PyTorch version setup is recorded against official instructions [12].")
        p.heading("AI assistance")
        p.text("Claude supplied initial notes. ChatGPT/Codex helped with coding, debugging, explanations, analysis, references, and report preparation. The user ran the cluster experiments and copied the outputs. The numbers here come from those saved outputs, not from published benchmark tables. The student remains responsible for reviewing the explanations and submission. This version changes the presentation, not the experiments or the evidence.",size=9.2)
        p.finish()

        p = Page(9, "References and further reading", pdf, md)
        p.text("Research papers [1–6] explain the original ideas and datasets. Official sources [7–10,12] describe the software used; [11] is the course tutorial. You do not need to read these to follow the main findings, but they let a reader check the technical explanations. Documentation was checked November 2, 2026.",size=9,gap=10)
        for r in references:
            p.text(f"[{r['id']}] {r['citation']}",size=8.5,gap=1)
            p.text(r['url'],size=7.3,gap=7,url=r['url'])
        p.finish()

    (output / "plain_language_report.md").write_text("\n".join(md))
    (output / "plain_language_sha256.json").write_text(json.dumps({
        pdf_path.name: hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    },indent=2)+"\n")
    print("Created submission/plain_language_report.pdf (9 pages) and Markdown source.")
    print("Previews:",layout.PREVIEW)


if __name__ == "__main__":
    main()
