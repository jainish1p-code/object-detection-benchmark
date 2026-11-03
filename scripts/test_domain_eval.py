"""Synthetic checks: python -m unittest discover -s scripts -p 'test_*.py'."""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from eval_coco import model_predictions, per_class_ap, sanity_predictions
from prepare_kitti import convert
from taxonomy import remap_coco


def coco_object(dataset):
    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()
    return coco


class DomainEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.training = self.root / "training"
        (self.training / "image_2").mkdir(parents=True)
        (self.training / "label_2").mkdir()
        for stem in ("000000", "000001"):
            Image.new("RGB", (200, 200)).save(self.training / "image_2" / f"{stem}.png")
        # Real objects at the top; a separate ignored region at the bottom.
        (self.training / "label_2/000000.txt").write_text(
            "Car 0 0 0 0 0 40 40 1 1 1 0 0 0 0\n"
            "Pedestrian 0 0 0 60 0 80 40 1 1 1 0 0 0 0\n"
            "DontCare -1 -1 -10 0 100 100 200 -1 -1 -1 -1000 -1000 -1000 -10\n"
        )
        (self.training / "label_2/000001.txt").write_text("")

    def test_conversion_and_ignore_matching(self):
        dataset = convert(self.training)
        self.assertEqual(len(dataset["images"]), 2)  # Keep negative images.
        self.assertEqual(len(dataset["annotations"]), 4)
        self.assertEqual(sum(a["iscrowd"] for a in dataset["annotations"]), 2)
        self.assertEqual(dataset["info"]["source_class_counts"]["Car"], 1)
        with contextlib.redirect_stdout(io.StringIO()):
            gt = coco_object(remap_coco(dataset))
            predictions = sanity_predictions(gt, [0, 1])
            self.assertEqual(len(predictions), 2)  # Ignore regions aren't predictions.
            # Higher-scored detections fully inside the ignore region must not hurt AP.
            for category_id in (1, 2):
                predictions.append({"image_id": 0, "category_id": category_id,
                                    "bbox": [10, 120, 20, 20], "score": 2.0})
            ev = COCOeval(gt, gt.loadRes(predictions), "bbox")
            ev.evaluate()
            ev.accumulate()
            self.assertEqual(per_class_ap(ev), {"vehicle": 1.0, "person": 1.0})
            # A false positive outside the ignore region must reduce AP.
            predictions.append({"image_id": 1, "category_id": 1,
                                "bbox": [100, 100, 20, 20], "score": 3.0})
            ev = COCOeval(gt, gt.loadRes(predictions), "bbox")
            ev.evaluate()
            ev.accumulate()
            self.assertLess(per_class_ap(ev)["vehicle"], 1.0)

    def test_both_ground_truth_and_predictions_are_mapped(self):
        dataset = {
            "images": [{"id": 1, "file_name": "000000.png", "width": 200, "height": 200}],
            "categories": [{"id": 10, "name": "car"}, {"id": 20, "name": "truck"},
                           {"id": 30, "name": "person"}, {"id": 40, "name": "dog"}],
            "annotations": [
                {"id": i, "image_id": 1, "category_id": cid, "bbox": [0, 0, 10, 10],
                 "area": 100, "iscrowd": 0}
                for i, cid in enumerate((10, 20, 30, 40), 1)
            ],
        }
        mapped = remap_coco(dataset)
        self.assertEqual([a["category_id"] for a in mapped["annotations"]], [1, 1, 2])
        self.assertEqual(len(dataset["annotations"]), 4)  # Native data is untouched.

        class FakeDetector:
            def predict(self, image):
                return [[0, 0, 10, 10]] * 4, [0.9] * 4, ["car", "truck", "person", "dog"]

        with contextlib.redirect_stdout(io.StringIO()):
            predictions, skipped = model_predictions(
                FakeDetector(), coco_object(mapped), [1], self.training / "image_2", "vehicle_person"
            )
        self.assertEqual([p["category_id"] for p in predictions], [1, 1, 2])
        self.assertEqual(skipped, {"dog": 1})

    def test_zero_predictions_are_saved_as_zero_ap(self):
        dataset = convert(self.training)
        # Force zero predictions even though both target classes have valid GT.
        ann_path = self.root / "gt.json"
        ann_path.write_text(json.dumps(dataset))
        out_path = self.root / "empty.json"
        code = (
            "import sys; sys.path.insert(0, sys.argv.pop(1)); import eval_coco; "
            "eval_coco.sanity_predictions = lambda coco, ids: []; eval_coco.main()"
        )
        subprocess.run([
            sys.executable, "-c", code, str(SRC), "--sanity", "--taxonomy", "vehicle_person",
            "--ann", str(ann_path), "--out", str(out_path),
        ], check=True, capture_output=True, text=True)
        record = json.loads(out_path.read_text())
        self.assertEqual(record["stats"]["AP"], 0.0)
        self.assertEqual(record["per_class_ap"], {"vehicle": 0.0, "person": 0.0})
        self.assertEqual(record["dataset_info"]["taxonomy"], "vehicle_person")

    def test_reject_missing_labels_and_unknown_classes(self):
        (self.training / "label_2/000001.txt").rename(self.root / "saved_label.txt")
        with self.assertRaisesRegex(ValueError, "IDs do not match"):
            convert(self.training)
        (self.root / "saved_label.txt").rename(self.training / "label_2/000001.txt")
        (self.training / "label_2/000001.txt").write_text(
            "Unknown 0 0 0 0 0 10 10 1 1 1 0 0 0 0\n"
        )
        with self.assertRaisesRegex(ValueError, "unknown class"):
            convert(self.training)


if __name__ == "__main__":
    unittest.main()
