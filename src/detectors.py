"""Three detector families behind one interface.

predict(PIL.Image RGB) -> (boxes_xyxy, scores, class_names)
  boxes: list of [x1, y1, x2, y2] in ORIGINAL image pixels
"""
import torch
import torchvision.transforms.functional as TF


class Detector:
    family = "base"

    def predict(self, img):
        raise NotImplementedError

    def describe(self):
        """Settings that must be recorded for reproducibility."""
        return {}


class FasterRCNNDetector(Detector):
    family = "two-stage CNN"

    def __init__(self, device, score_thr, nms_thr=None):
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,
        )
        w = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        kw = {"box_score_thresh": score_thr}
        if nms_thr is not None:
            kw["box_nms_thresh"] = nms_thr
        self.model = fasterrcnn_resnet50_fpn_v2(weights=w, **kw).to(device).eval()
        self.categories = w.meta["categories"]
        self.device = device

    @torch.no_grad()
    def predict(self, img):
        x = TF.to_tensor(img).to(self.device)   # model resizes/normalizes internally
        out = self.model([x])[0]
        names = [self.categories[i] for i in out["labels"].tolist()]
        return out["boxes"].cpu().tolist(), out["scores"].cpu().tolist(), names

    def describe(self):
        m = self.model
        return {"model": "fasterrcnn_resnet50_fpn_v2", "weights": "COCO_V1",
                "score_thr": m.roi_heads.score_thresh, "nms_thr": m.roi_heads.nms_thresh,
                "max_dets": m.roi_heads.detections_per_img,
                "input": f"resized internally, min_side={m.transform.min_size}, max_side={m.transform.max_size}"}


class YOLODetector(Detector):
    family = "one-stage CNN"

    def __init__(self, ckpt, device, score_thr, nms_thr=None, imgsz=640):
        from ultralytics import YOLO
        self.ckpt, self.device, self.imgsz = ckpt, device, imgsz
        self.conf = score_thr
        self.iou = 0.7 if nms_thr is None else nms_thr
        self.model = YOLO(ckpt)

    def predict(self, img):
        r = self.model.predict(img, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                               device=self.device, max_det=100, verbose=False)[0]
        names = [r.names[int(c)] for c in r.boxes.cls.tolist()]
        return r.boxes.xyxy.cpu().tolist(), r.boxes.conf.cpu().tolist(), names

    def describe(self):
        return {"model": self.ckpt, "score_thr": self.conf, "nms_thr": self.iou,
                "max_dets": 100, "input": f"letterboxed to imgsz={self.imgsz}"}


class DETRDetector(Detector):
    family = "transformer"

    def __init__(self, ckpt, device, score_thr):
        from transformers import DetrForObjectDetection, DetrImageProcessor
        self.ckpt, self.device, self.thr = ckpt, device, score_thr
        self.processor = DetrImageProcessor.from_pretrained(ckpt)
        self.model = DetrForObjectDetection.from_pretrained(ckpt).to(device).eval()

    @torch.no_grad()
    def predict(self, img):
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        res = self.processor.post_process_object_detection(
            out, threshold=self.thr, target_sizes=[(img.height, img.width)])[0]
        id2label = self.model.config.id2label
        names = [id2label[i] for i in res["labels"].tolist()]
        return res["boxes"].cpu().tolist(), res["scores"].cpu().tolist(), names

    def describe(self):
        return {"model": self.ckpt, "score_thr": self.thr, "nms_thr": None,
                "max_dets": 100, "input": "resized by DetrImageProcessor (shortest side 800, longest <= 1333)"}


def build_detector(name, device, score_thr, nms_thr=None, imgsz=640):
    if name == "frcnn":
        return FasterRCNNDetector(device, score_thr, nms_thr)
    if name.startswith("yolo"):
        return YOLODetector(name if name.endswith(".pt") else name + ".pt",
                            device, score_thr, nms_thr, imgsz)
    if name == "detr":
        return DETRDetector("facebook/detr-resnet-50", device, score_thr)
    raise ValueError(f"unknown model {name!r}: use frcnn, yolo11s, detr")