import json, sys, urllib.request
from pathlib import Path

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
root = Path("data/coco")
ann = json.load(open(root / "annotations/instances_val2017.json"))

images = sorted(ann["images"], key=lambda im: im["id"])[:N]
ids = {im["id"] for im in images}
mini = {
    "images": images,
    "annotations": [a for a in ann["annotations"] if a["image_id"] in ids],
    "categories": ann["categories"],
}
json.dump(mini, open(root / "mini_val.json", "w"))

img_dir = root / "val2017"
img_dir.mkdir(exist_ok=True)
for im in images:
    dst = img_dir / im["file_name"]
    if not dst.exists():
        urllib.request.urlretrieve(im["coco_url"], dst)

print(len(images), "images,", len(mini["annotations"]), "annotations")
print("first annotation:", mini["annotations"][0])
print("first category:", mini["categories"][0])