# Recorded experiment commands

Run from the cluster repository root.
These are the original commands, including their original output paths.

## results/50149631/frcnn_eval.json

```bash
/ocean/projects/cis260281p/jpatel17/envs/det/bin/python src/eval_coco.py --model frcnn --device cuda --ann data/coco/annotations/instances_val2017.json --img-dir data/coco/val2017 --max-images 0 --score-thr 0.001 --out results/coco/50149631/frcnn_eval.json
```

## results/50151110/coco_frcnn_eval.json

```bash
/ocean/projects/cis260281p/jpatel17/envs/det/bin/python src/eval_coco.py --model frcnn --device cuda --score-thr 0.001 --ann data/coco/annotations/instances_val2017.json --img-dir data/coco/val2017 --taxonomy vehicle_person --out results/domains/50151110/coco_frcnn_eval.json
```

## results/50151110/kitti_frcnn_eval.json

```bash
/ocean/projects/cis260281p/jpatel17/envs/det/bin/python src/eval_coco.py --model frcnn --device cuda --score-thr 0.001 --ann data/kitti/annotations/vehicle_person_train.json --img-dir data/kitti/training/image_2 --taxonomy vehicle_person --out results/domains/50151110/kitti_frcnn_eval.json
```

## results/50149631/frcnn_benchmark.json

```bash
/ocean/projects/cis260281p/jpatel17/envs/det/bin/python src/benchmark.py --model frcnn --device cuda --ann data/coco/annotations/instances_val2017.json --img-dir data/coco/val2017 --max-images 100 --warmup 20 --iterations 100 --score-thr 0.001 --out results/coco/50149631/frcnn_benchmark.json
```

## results/50151110/kitti_frcnn_benchmark.json

```bash
/ocean/projects/cis260281p/jpatel17/envs/det/bin/python src/benchmark.py --model frcnn --device cuda --score-thr 0.001 --ann data/kitti/annotations/vehicle_person_train.json --img-dir data/kitti/training/image_2 --max-images 100 --warmup 20 --iterations 100 --out results/domains/50151110/kitti_frcnn_benchmark.json
```

## results/50149631/yolo11s_eval.json

```bash
/ocean/projects/cis260281p/jpatel17/envs/det/bin/python src/eval_coco.py --model yolo11s --device cuda --ann data/coco/annotations/instances_val2017.json --img-dir data/coco/val2017 --max-images 0 --score-thr 0.001 --out results/coco/50149631/yolo11s_eval.json
```

## results/50151110/coco_yolo11s_eval.json

```bash
/ocean/projects/cis260281p/jpatel17/envs/det/bin/python src/eval_coco.py --model yolo11s --device cuda --score-thr 0.001 --ann data/coco/annotations/instances_val2017.json --img-dir data/coco/val2017 --taxonomy vehicle_person --out results/domains/50151110/coco_yolo11s_eval.json
```

## results/50151110/kitti_yolo11s_eval.json

```bash
/ocean/projects/cis260281p/jpatel17/envs/det/bin/python src/eval_coco.py --model yolo11s --device cuda --score-thr 0.001 --ann data/kitti/annotations/vehicle_person_train.json --img-dir data/kitti/training/image_2 --taxonomy vehicle_person --out results/domains/50151110/kitti_yolo11s_eval.json
```

## results/50149631/yolo11s_benchmark.json

```bash
/ocean/projects/cis260281p/jpatel17/envs/det/bin/python src/benchmark.py --model yolo11s --device cuda --ann data/coco/annotations/instances_val2017.json --img-dir data/coco/val2017 --max-images 100 --warmup 20 --iterations 100 --score-thr 0.001 --out results/coco/50149631/yolo11s_benchmark.json
```

## results/50151110/kitti_yolo11s_benchmark.json

```bash
/ocean/projects/cis260281p/jpatel17/envs/det/bin/python src/benchmark.py --model yolo11s --device cuda --score-thr 0.001 --ann data/kitti/annotations/vehicle_person_train.json --img-dir data/kitti/training/image_2 --max-images 100 --warmup 20 --iterations 100 --out results/domains/50151110/kitti_yolo11s_benchmark.json
```

## results/50149631/detr_eval.json

```bash
/ocean/projects/cis260281p/jpatel17/envs/det/bin/python src/eval_coco.py --model detr --device cuda --ann data/coco/annotations/instances_val2017.json --img-dir data/coco/val2017 --max-images 0 --score-thr 0.001 --out results/coco/50149631/detr_eval.json
```

## results/50151110/coco_detr_eval.json

```bash
/ocean/projects/cis260281p/jpatel17/envs/det/bin/python src/eval_coco.py --model detr --device cuda --score-thr 0.001 --ann data/coco/annotations/instances_val2017.json --img-dir data/coco/val2017 --taxonomy vehicle_person --out results/domains/50151110/coco_detr_eval.json
```

## results/50151110/kitti_detr_eval.json

```bash
/ocean/projects/cis260281p/jpatel17/envs/det/bin/python src/eval_coco.py --model detr --device cuda --score-thr 0.001 --ann data/kitti/annotations/vehicle_person_train.json --img-dir data/kitti/training/image_2 --taxonomy vehicle_person --out results/domains/50151110/kitti_detr_eval.json
```

## results/50149631/detr_benchmark.json

```bash
/ocean/projects/cis260281p/jpatel17/envs/det/bin/python src/benchmark.py --model detr --device cuda --ann data/coco/annotations/instances_val2017.json --img-dir data/coco/val2017 --max-images 100 --warmup 20 --iterations 100 --score-thr 0.001 --out results/coco/50149631/detr_benchmark.json
```

## results/50151110/kitti_detr_benchmark.json

```bash
/ocean/projects/cis260281p/jpatel17/envs/det/bin/python src/benchmark.py --model detr --device cuda --score-thr 0.001 --ann data/kitti/annotations/vehicle_person_train.json --img-dir data/kitti/training/image_2 --max-images 100 --warmup 20 --iterations 100 --out results/domains/50151110/kitti_detr_benchmark.json
```

## results/50151110/coco_sanity.json

```bash
/ocean/projects/cis260281p/jpatel17/envs/det/bin/python src/eval_coco.py --sanity --ann data/coco/annotations/instances_val2017.json --img-dir data/coco/val2017 --taxonomy vehicle_person --out results/domains/50151110/coco_sanity.json
```

## results/50151110/kitti_sanity.json

```bash
/ocean/projects/cis260281p/jpatel17/envs/det/bin/python src/eval_coco.py --sanity --ann data/kitti/annotations/vehicle_person_train.json --img-dir data/kitti/training/image_2 --taxonomy vehicle_person --out results/domains/50151110/kitti_sanity.json
```

## KITTI annotation conversion

```bash
/ocean/projects/cis260281p/jpatel17/envs/det/bin/python src/prepare_kitti.py
```
