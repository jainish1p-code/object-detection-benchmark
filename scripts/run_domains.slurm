#!/bin/bash
# Submit from the repository root after uploading source files.
#SBATCH --job-name=coco-kitti
#SBATCH --account=cis260281p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=domains_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
PYTHON_BIN=/ocean/projects/cis260281p/jpatel17/envs/det/bin/python
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
OUT_DIR="results/domains/$SLURM_JOB_ID"
mkdir -p "$OUT_DIR"
date
hostname
nvidia-smi
"$PYTHON_BIN" -m pip freeze > "$OUT_DIR/packages.txt"

"$PYTHON_BIN" src/prepare_kitti.py

for dataset in coco kitti; do
    if [[ "$dataset" == coco ]]; then
        ANN=data/coco/annotations/instances_val2017.json
        IMG_DIR=data/coco/val2017
    else
        ANN=data/kitti/annotations/vehicle_person_train.json
        IMG_DIR=data/kitti/training/image_2
    fi
    # Check conversion and ground-truth remapping before loading models.
    "$PYTHON_BIN" src/eval_coco.py --sanity \
        --ann "$ANN" --img-dir "$IMG_DIR" --taxonomy vehicle_person \
        --out "$OUT_DIR/${dataset}_sanity.json"

    for model in frcnn yolo11s detr; do
        echo "Evaluating $model on $dataset with the shared vehicle/person taxonomy"
        "$PYTHON_BIN" src/eval_coco.py \
            --model "$model" --device cuda --score-thr 0.001 \
            --ann "$ANN" --img-dir "$IMG_DIR" --taxonomy vehicle_person \
            --out "$OUT_DIR/${dataset}_${model}_eval.json"

        # Native-COCO timing is already saved; add the KITTI measurements.
        if [[ "$dataset" == kitti ]]; then
            "$PYTHON_BIN" src/benchmark.py \
                --model "$model" --device cuda --score-thr 0.001 \
                --ann "$ANN" --img-dir "$IMG_DIR" --max-images 100 \
                --warmup 20 --iterations 100 \
                --out "$OUT_DIR/kitti_${model}_benchmark.json"
        fi
    done
done
echo "Completed COCO/KITTI shared-taxonomy evaluation; results in $OUT_DIR"
date
