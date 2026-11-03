#!/bin/bash
# Submit from the repository root: sbatch scripts/run_coco.slurm
#SBATCH --job-name=coco-benchmark
#SBATCH --account=cis260281p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=coco_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

# Use the homework environment directly, even if submitted from (base).
PYTHON_BIN=/ocean/projects/cis260281p/jpatel17/envs/det/bin/python
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
ANN=data/coco/annotations/instances_val2017.json
IMG_DIR=data/coco/val2017
# A job-specific directory preserves results from any earlier submission.
OUT_DIR="results/coco/$SLURM_JOB_ID"
mkdir -p "$OUT_DIR"

date
hostname
nvidia-smi
"$PYTHON_BIN" -m pip freeze > "$OUT_DIR/packages.txt"

for model in frcnn yolo11s detr; do
    echo "Evaluating $model on all 5,000 COCO validation images"
    "$PYTHON_BIN" src/eval_coco.py \
        --model "$model" --device cuda \
        --ann "$ANN" --img-dir "$IMG_DIR" --max-images 0 \
        --score-thr 0.001 --out "$OUT_DIR/${model}_eval.json"

    echo "Benchmarking $model: 20 warmup and 100 measured predictions"
    "$PYTHON_BIN" src/benchmark.py \
        --model "$model" --device cuda \
        --ann "$ANN" --img-dir "$IMG_DIR" --max-images 100 \
        --warmup 20 --iterations 100 --score-thr 0.001 \
        --out "$OUT_DIR/${model}_benchmark.json"
done

echo "Completed all three models; results saved to $OUT_DIR"
date
