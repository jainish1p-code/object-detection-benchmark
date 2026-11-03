"""Batch-size-1 predict() latency and PyTorch peak CUDA memory.

Run each model in a separate process, for example:
  python src/benchmark.py --model frcnn --device cuda:0
"""
import argparse
import json
import platform
import shlex
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from detectors import build_detector


def pkg(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def measure(det, image_paths, device, warmup, iterations):
    """Cycle through images; exclude file loading and warmup from latency."""
    if not image_paths or warmup < 1 or iterations < 1:
        raise ValueError("need images and positive warmup/iterations")
    device = torch.device(device)
    use_cuda = device.type == "cuda"

    def synchronize():
        if use_cuda:
            torch.cuda.synchronize(device)

    def load_image(index):
        with Image.open(image_paths[index % len(image_paths)]) as image:
            return image.convert("RGB")

    with torch.inference_mode():
        # Initialize kernels, caches, and lazy model setup before measuring.
        for i in range(warmup):
            det.predict(load_image(i))
        synchronize()
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(device)

        samples_ms = []
        for i in range(iterations):
            image = load_image(i)  # Disk access and decoding are not timed.
            synchronize()
            start = time.perf_counter()
            det.predict(image)
            synchronize()  # Wait for asynchronous CUDA work before stopping.
            samples_ms.append((time.perf_counter() - start) * 1000)

    # Peak allocated includes model weights and live tensors, not just activations.
    # Reserved also includes memory held by PyTorch's CUDA caching allocator.
    memory = {
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2 if use_cuda else None
        ),
        "peak_reserved_mib": (
            torch.cuda.max_memory_reserved(device) / 1024**2 if use_cuda else None
        ),
    }
    mean_ms = float(np.mean(samples_ms))
    return {
        "latency_ms": {
            "mean": mean_ms,
            "p50": float(np.percentile(samples_ms, 50)),
            "p95": float(np.percentile(samples_ms, 95)),
        },
        "fps": 1000 / mean_ms,
        "gpu_memory": memory,
        "samples_ms": samples_ms,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="frcnn", help="frcnn | yolo11s | detr")
    p.add_argument("--ann", default="data/coco/mini_val.json")
    p.add_argument("--img-dir", default="data/coco/val2017")
    p.add_argument("--max-images", type=int, default=20, help="0 = all; cycle in sorted ID order")
    p.add_argument("--warmup", type=int, default=10, help="untimed predictions")
    p.add_argument("--iterations", type=int, default=100, help="timed predictions")
    p.add_argument("--score-thr", type=float, default=0.001)
    p.add_argument("--nms-thr", type=float, default=None, help="default: each family's own")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO only")
    p.add_argument("--device", default="cpu", help="cpu | cuda | cuda:N")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.warmup < 1 or args.iterations < 1 or args.max_images < 0:
        p.error("warmup and iterations must be positive; max-images must be nonnegative")
    try:
        device = torch.device(args.device)
    except (RuntimeError, ValueError) as error:
        p.error(str(error))
    if device.type not in {"cpu", "cuda"}:
        p.error("supported devices: cpu, cuda, cuda:N")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            p.error("CUDA is unavailable; use --device cpu for a local smoke test")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device.index >= torch.cuda.device_count():
            p.error(f"CUDA device index {device.index} is unavailable")
        # Also ensures libraries that use the current device use the requested GPU.
        torch.cuda.set_device(device)

    with open(args.ann) as file:
        images = sorted(json.load(file)["images"], key=lambda image: image["id"])
    if args.max_images:
        images = images[:args.max_images]
    if not images:
        p.error("annotation file contains no images")
    image_paths = [Path(args.img_dir) / image["file_name"] for image in images]
    for path in image_paths:
        if not path.is_file():
            p.error(f"image not found: {path}")

    det = build_detector(args.model, str(device), args.score_thr, args.nms_thr, args.imgsz)
    print(f"{args.model} on {device}: {args.warmup} warmup, {args.iterations} timed predictions",
          flush=True)
    measurements = measure(det, image_paths, device, args.warmup, args.iterations)
    record = {
        "model": args.model,
        "detector": det.describe(),
        "args": vars(args),
        "command": shlex.join([sys.executable, *sys.argv]),
        "cwd": str(Path.cwd()),
        "protocol": {
            "batch_size": 1,
            "timing": "predict(): preprocessing, device transfers, inference, postprocessing, CPU outputs",
            "excluded": "model loading, image file loading/decoding, warmup",
            "image_order": "sorted image ID, repeated cyclically; timed loop restarts at first image",
            "cuda_synchronize": device.type == "cuda",
            "memory": "PyTorch allocator peak after warmup; includes resident weights; excludes non-PyTorch allocations",
            "fps": "1000 / mean latency in ms; serial batch-size-1 throughput",
        },
        "num_images": len(images),
        "images": images,
        "env": {
            "python": platform.python_version(),
            "machine": platform.platform(),
            "cpu": platform.processor() or platform.machine(),
            "device": str(device),
            "torch": torch.__version__,
            "torchvision": pkg("torchvision"),
            "ultralytics": pkg("ultralytics"),
            "transformers": pkg("transformers"),
            "numpy": pkg("numpy"),
            "pillow": pkg("pillow"),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if device.type == "cuda" else None,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch_num_threads": torch.get_num_threads(),
        },
        **measurements,
    }
    out_path = Path(args.out or f"results/smoke/{args.model}_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as file:
        json.dump(record, file, indent=2)
    latency = measurements["latency_ms"]
    print(f"Latency (ms): mean={latency['mean']:.2f}, p50={latency['p50']:.2f}, p95={latency['p95']:.2f}")
    print(f"FPS: {measurements['fps']:.2f}")
    if device.type == "cuda":
        memory = measurements["gpu_memory"]
        print(f"Peak GPU memory (MiB): allocated={memory['peak_allocated_mib']:.2f}, "
              f"reserved={memory['peak_reserved_mib']:.2f}")
    else:
        print("Peak GPU memory: unavailable on CPU (saved as null)")
    print("saved", out_path)


if __name__ == "__main__":
    main()
