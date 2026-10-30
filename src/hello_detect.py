import torch
from torchvision.io import read_image
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5).eval()
preprocess = weights.transforms()
categories = weights.meta["categories"]

img = read_image("data/sample.jpg")

with torch.no_grad():
    out = model([preprocess(img)])[0]

for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
    print(categories[label], round(score.item(), 3), [round(v) for v in box.tolist()])