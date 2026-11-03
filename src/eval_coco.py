import argparse, json, platform, shlex, sys, time
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from detectors import build_detector
from taxonomy import COCO_TO_SHARED, remap_coco, shared_name

STAT_NAMES = [
    "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large",
]


def pkg(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def sanity_predictions(coco, img_ids):
    """Feed ground truth back in as 'predictions'. Should score ~1.0."""
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_ids))
    return [{"image_id": a["image_id"], "category_id": a["category_id"],
             "bbox": a["bbox"], "score": 1.0} for a in anns if not a.get("iscrowd", 0)]


def per_class_ap(ev):
    """Average over IoUs and recall, using all areas and maxDets=100."""
    # COCO precision axes: IoU, recall, category, area, max detections.
    precision = ev.eval["precision"]
    values = {}
    for index, category_id in enumerate(ev.params.catIds):
        category = ev.cocoGt.loadCats([category_id])[0]
        scores = precision[:, :, index, 0, -1]
        scores = scores[scores > -1]  # -1 means no evaluable ground truth.
        values[category["name"]] = round(float(scores.mean()), 4) if scores.size else None
    return values


def model_predictions(det, coco, img_ids, img_dir, taxonomy="native"):
    name_to_id = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}
    skipped = {}
    results = []
    for img_id in tqdm(img_ids):
        info = coco.loadImgs(img_id)[0]
        with Image.open(Path(img_dir) / info["file_name"]) as source:
            img = source.convert("RGB")
        boxes, scores, names = det.predict(img)
        for (x1, y1, x2, y2), s, n in zip(boxes, scores, names):
            mapped_name = shared_name(n) if taxonomy == "vehicle_person" else n
            cid = name_to_id.get(mapped_name)
            if cid is None:                      # class not in this dataset's label set
                skipped[n] = skipped.get(n, 0) + 1
                continue
            results.append({
                "image_id": img_id, "category_id": cid,
                "bbox": [round(x1, 2), round(y1, 2),
                         round(x2 - x1, 2), round(y2 - y1, 2)],   # xyxy -> xywh
                "score": round(s, 4),
            })
    return results, skipped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="frcnn", help="frcnn | yolo11s | detr")
    p.add_argument("--ann", default="data/coco/mini_val.json")
    p.add_argument("--img-dir", default="data/coco/val2017")
    p.add_argument("--max-images", type=int, default=0, help="0 = all")
    p.add_argument("--score-thr", type=float, default=0.001)
    p.add_argument("--nms-thr", type=float, default=None, help="default: each family's own")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO only")
    p.add_argument("--device", default="cpu")
    p.add_argument("--taxonomy", choices=["native", "vehicle_person"], default="native")
    p.add_argument("--sanity", action="store_true", help="use GT as predictions")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out_path = args.out or f"results/smoke/{'sanity' if args.sanity else args.model}_mini.json"

    coco = COCO(args.ann)
    source_info = coco.dataset.get("info", {})
    if source_info.get("taxonomy") == "vehicle_person" and args.taxonomy != "vehicle_person":
        p.error("these annotations require --taxonomy vehicle_person to map detector outputs")
    if args.taxonomy == "vehicle_person":
        coco.dataset = remap_coco(coco.dataset)
        coco.createIndex()
    img_ids = sorted(coco.getImgIds())
    if args.max_images < 0:
        p.error("max-images must be nonnegative")
    if args.max_images:
        img_ids = img_ids[: args.max_images]
    if not img_ids:
        p.error("annotation file contains no images")

    t0 = time.time()
    det, skipped = None, {}
    if args.sanity:
        results = sanity_predictions(coco, img_ids)
    else:
        det = build_detector(args.model, args.device, args.score_thr, args.nms_thr, args.imgsz)
        results, skipped = model_predictions(det, coco, img_ids, args.img_dir, args.taxonomy)
    print(f"{len(results)} detections on {len(img_ids)} images; skipped (unmapped classes): {skipped}")
    if results:
        coco_dt = coco.loadRes(results)
    else:
        # A detector with no mapped predictions still has valid (zero) AP/AR.
        coco_dt = COCO()
        coco_dt.dataset = {"images": coco.dataset["images"],
                           "categories": coco.dataset["categories"], "annotations": []}
        coco_dt.createIndex()
    ev = COCOeval(coco, coco_dt, "bbox")
    ev.params.imgIds = img_ids
    ev.evaluate(); ev.accumulate(); ev.summarize()

    record = {
        "mode": "sanity_gt" if args.sanity else args.model,
        "detector": det.describe() if det else None,
        "args": vars(args),
        "command": shlex.join([sys.executable, *sys.argv]),
        "cwd": str(Path.cwd()),
        "num_images": len(img_ids),
        "image_ids": img_ids,
        "dataset_info": source_info,
        "taxonomy": {"name": args.taxonomy,
                     "prediction_mapping": COCO_TO_SHARED if args.taxonomy == "vehicle_person" else None,
                     "categories": coco.dataset["categories"]},
        "num_detections": len(results),
        "skipped_unmapped": skipped,
        "stats": dict(zip(STAT_NAMES, [round(float(s), 4) for s in ev.stats])),
        "per_class_ap": per_class_ap(ev),
        "env": {"python": platform.python_version(), "machine": platform.platform(),
                "torch": torch.__version__, "torchvision": pkg("torchvision"),
                "ultralytics": pkg("ultralytics"), "transformers": pkg("transformers"),
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "wall_seconds": round(time.time() - t0, 1),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(record, open(out_path, "w"), indent=2)
    print("saved", out_path)


if __name__ == "__main__":
    main()
