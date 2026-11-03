"""Convert KITTI's labeled training images to a two-class COCO-style dataset."""
import argparse
import json
import math
import shlex
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

from taxonomy import KITTI_IGNORED, KITTI_TO_SHARED, SHARED_CATEGORIES


def convert(training_dir):
    training_dir = Path(training_dir)
    image_dir = training_dir / "image_2"
    label_dir = training_dir / "label_2"
    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise ValueError(f"no PNG images in {image_dir}")
    image_ids = {path.stem for path in image_paths}
    label_ids = {path.stem for path in label_dir.glob("*.txt")}
    if image_ids != label_ids:
        raise ValueError("image/label IDs do not match; finish extraction before converting")

    images, annotations = [], []
    counts = Counter()
    name_to_id = {category["name"]: category["id"] for category in SHARED_CATEGORIES}
    for path in image_paths:
        image_id = int(path.stem)
        with Image.open(path) as image:
            width, height = image.size
        images.append({"id": image_id, "file_name": path.name,
                       "width": width, "height": height})
        label_path = label_dir / f"{path.stem}.txt"
        for line_number, line in enumerate(label_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) < 15:
                raise ValueError(f"{label_path}:{line_number}: expected at least 15 fields")
            name = fields[0]
            if name not in KITTI_TO_SHARED and name not in KITTI_IGNORED:
                raise ValueError(f"{label_path}:{line_number}: unknown class {name!r}")
            x1, y1, x2, y2 = map(float, fields[4:8])
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                raise ValueError(f"{label_path}:{line_number}: non-finite box")
            x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
            y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"{label_path}:{line_number}: empty box after clipping")
            counts[name] += 1
            ignored = name in KITTI_IGNORED
            # A region is ignored for BOTH evaluated classes. COCO's crowd matching
            # ignores overlapping detections, while real ground truth takes priority.
            category_ids = list(name_to_id.values()) if ignored else [name_to_id[KITTI_TO_SHARED[name]]]
            for category_id in category_ids:
                annotations.append({
                    "id": len(annotations) + 1, "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1), "iscrowd": int(ignored),
                    "source_class": name,
                })

    return {
        "info": {
            "dataset": "KITTI object detection", "split": "all labeled training images; no retraining",
            "taxonomy": "vehicle_person", "label_mapping": KITTI_TO_SHARED,
            "source_class_counts": dict(sorted(counts.items())),
            "evaluation_protocol": {
                "metric": "COCO-style bbox AP/AR, not official KITTI AP_R40",
                "ignore": "Cyclist/Misc/DontCare boxes become iscrowd=1 regions for both classes; "
                          "COCO matching uses intersection/detection-area at each IoU threshold",
                "difficulty": "all annotated target objects; no easy/moderate/hard filtering",
                "boxes": "xyxy converted to xywh; clipped to image bounds; no +1 pixel adjustment",
                "limitation": "shared labels approximate different annotation taxonomies; "
                              "ignored regions can also overlap real target objects",
            },
        },
        "images": images, "annotations": annotations, "categories": SHARED_CATEGORIES,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", default="data/kitti/training")
    parser.add_argument("--out", default="data/kitti/annotations/vehicle_person_train.json")
    args = parser.parse_args()
    dataset = convert(args.training_dir)
    dataset["info"]["conversion_command"] = shlex.join([sys.executable, *sys.argv])
    dataset["info"]["cwd"] = str(Path.cwd())
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as file:
        json.dump(dataset, file)
    print("Images:", len(dataset["images"]))
    print("Source object counts:", dataset["info"]["source_class_counts"])
    print("Saved", out_path)


if __name__ == "__main__":
    main()
