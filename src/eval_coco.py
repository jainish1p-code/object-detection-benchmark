import argparse, json, platform, time
from pathlib import Path

import torch, torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.io import read_image, ImageReadMode
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,
)
from tqdm import tqdm

STAT_NAMES = [
    "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large",
]


def sanity_predictions(coco, img_ids):
    """Feed ground truth back in as 'predictions'. Should score ~1.0."""
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_ids))
    return [{"image_id": a["image_id"], "category_id": a["category_id"],
             "bbox": a["bbox"], "score": 1.0} for a in anns]


def model_predictions(coco, img_ids, img_dir, device, score_thr):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=score_thr)
    model = model.to(device).eval()
    preprocess = weights.transforms()

    results = []
    for img_id in tqdm(img_ids):
        info = coco.loadImgs(img_id)[0]
        # mode=RGB matters: some COCO images are grayscale
        img = read_image(str(Path(img_dir) / info["file_name"]), mode=ImageReadMode.RGB)
        with torch.no_grad():
            out = model([preprocess(img).to(device)])[0]
        for box, label, score in zip(out["boxes"].tolist(),
                                     out["labels"].tolist(),
                                     out["scores"].tolist()):
            x1, y1, x2, y2 = box
            results.append({
                "image_id": img_id,
                "category_id": int(label),                       # already a COCO id
                "bbox": [round(x1, 2), round(y1, 2),
                         round(x2 - x1, 2), round(y2 - y1, 2)],  # xyxy -> xywh
                "score": round(score, 4),
            })
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ann", default="data/coco/mini_val.json")
    p.add_argument("--img-dir", default="data/coco/val2017")
    p.add_argument("--max-images", type=int, default=0, help="0 = all")
    p.add_argument("--score-thr", type=float, default=0.001)
    p.add_argument("--device", default="cpu")
    p.add_argument("--sanity", action="store_true", help="use GT as predictions")
    p.add_argument("--out", default="results/smoke/frcnn_mini.json")
    args = p.parse_args()

    coco = COCO(args.ann)
    img_ids = sorted(coco.getImgIds())
    if args.max_images:
        img_ids = img_ids[: args.max_images]

    t0 = time.time()
    if args.sanity:
        results = sanity_predictions(coco, img_ids)
    else:
        results = model_predictions(coco, img_ids, args.img_dir, args.device, args.score_thr)
    print(f"{len(results)} detections on {len(img_ids)} images")

    coco_dt = coco.loadRes(results)
    ev = COCOeval(coco, coco_dt, "bbox")
    ev.params.imgIds = img_ids          # only score the images we actually ran
    ev.evaluate(); ev.accumulate(); ev.summarize()

    record = {
        "mode": "sanity_gt" if args.sanity else "faster_rcnn_resnet50_fpn_v2",
        "args": vars(args),
        "num_images": len(img_ids),
        "num_detections": len(results),
        "stats": dict(zip(STAT_NAMES, [round(float(s), 4) for s in ev.stats])),
        "env": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                "python": platform.python_version(), "machine": platform.platform()},
        "wall_seconds": round(time.time() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(record, open(args.out, "w"), indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()