"""Export lightweight paper metrics for a NEU-DET RT-DETR run."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.core import YAMLConfig  # noqa: E402


CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--measure-speed", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-size", type=int, nargs=2, default=[640, 640], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    return parser.parse_args()


def _mean_valid(values: np.ndarray) -> float:
    values = values[values > -1]
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _index_or_default(items, value, default=-1) -> int:
    try:
        return list(items).index(value)
    except ValueError:
        return default


def _iou_index(iou_thrs: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(iou_thrs - value)))


def summarize_eval(eval_data: dict) -> tuple[dict, dict[str, float]]:
    precision = np.asarray(eval_data["precision"])
    params = eval_data["params"]

    area_idx = _index_or_default(params.areaRngLbl, "all", 0)
    max_det_idx = _index_or_default(params.maxDets, 100, len(params.maxDets) - 1)
    map50_idx = _iou_index(np.asarray(params.iouThrs), 0.50)
    map75_idx = _iou_index(np.asarray(params.iouThrs), 0.75)

    all_precision = precision[:, :, :, area_idx, max_det_idx]
    map50_95 = _mean_valid(all_precision)
    map50 = _mean_valid(precision[map50_idx, :, :, area_idx, max_det_idx])
    map75 = _mean_valid(precision[map75_idx, :, :, area_idx, max_det_idx])

    ap_per_class = {}
    for idx, class_name in enumerate(CLASSES):
        if idx < precision.shape[2]:
            ap_per_class[class_name] = _mean_valid(precision[:, :, idx, area_idx, max_det_idx])
        else:
            ap_per_class[class_name] = 0.0

    metrics = {
        "map50_95": map50_95,
        "map50": map50,
        "map75": map75,
    }
    return metrics, ap_per_class


def state_dict_size_mb(state: dict) -> float:
    total_bytes = 0
    for value in state.values():
        if torch.is_tensor(value):
            total_bytes += value.numel() * value.element_size()
    return total_bytes / (1024 * 1024)


def compute_gflops(model: torch.nn.Module, images: torch.Tensor) -> tuple[float, float, str]:
    try:
        from thop import profile
    except Exception as exc:
        return 0.0, 0.0, f"thop unavailable: {exc}"

    try:
        macs, _ = profile(model, inputs=(images,), verbose=False)
    except Exception as exc:
        return 0.0, 0.0, f"thop profile failed: {exc}"
    gmacs = float(macs) / 1e9
    return gmacs * 2.0, gmacs, ""


def measure_speed(config: Path, resume: Path, device_name: str, input_size: list[int], warmup: int, repeat: int) -> dict:
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    cfg = YAMLConfig(
        str(config),
        resume=str(resume),
        PResNet={"pretrained": False},
        val_dataloader={"batch_size": 1, "num_workers": 0},
    )
    checkpoint = torch.load(resume, map_location="cpu")
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)
    params_m = sum(p.numel() for p in cfg.model.parameters() if p.requires_grad) / 1e6

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = Model().to(device).eval()

    height, width = input_size
    dummy_images = torch.randn(1, 3, height, width, device=device)
    gflops, gmacs, gflops_note = compute_gflops(model.model, dummy_images)

    data_loader = cfg.val_dataloader
    warmup_batch = None
    for samples, targets in data_loader:
        warmup_batch = (
            samples.to(device),
            torch.stack([t["orig_size"] for t in targets], dim=0).to(device),
        )
        break
    if warmup_batch is None:
        raise RuntimeError("No samples found for speed measurement")

    with torch.no_grad():
        for _ in range(warmup):
            model(*warmup_batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        frame_count = 0
        start = time.perf_counter()
        for samples, targets in data_loader:
            samples = samples.to(device)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
            model(samples, orig_target_sizes)
            frame_count += samples.shape[0]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

    latency_ms = elapsed / max(frame_count, 1) * 1000.0
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
    metrics = {
        "params_m": params_m,
        "gflops": gflops,
        "gmacs": gmacs,
        "fps": fps,
        "latency_ms": latency_ms,
        "input_size": [height, width],
        "batch_size": 1,
        "warmup": warmup,
        "frame_count": frame_count,
        "device": str(device),
        "speed_protocol": "test2017 real images, batch_size=1, model+postprocessor, excludes dataloader/preprocess/COCOEval",
        "params_protocol": "trainable parameters before deploy(), matching training log",
        "gflops_protocol": "THOP MACs on deployed RT-DETR model with 1x3x640x640 input; report GFLOPs as 2 x GMACs",
    }
    if gflops_note:
        metrics["gflops_note"] = gflops_note
    return metrics


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_results_csv(path: Path, row: dict) -> None:
    fields = ["exp_name", "seed", "map50_95", "map50", "map75", "model_size_mb"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def write_per_class_csv(path: Path, ap_per_class: dict[str, float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "ap"])
        writer.writeheader()
        for class_name in CLASSES:
            writer.writerow({"class": class_name, "ap": ap_per_class[class_name]})


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    eval_path = output_dir / "eval.pth"
    if not eval_path.is_file():
        raise FileNotFoundError(f"Missing eval file: {eval_path}")
    if not args.resume.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.resume}")

    output_dir.mkdir(parents=True, exist_ok=True)
    eval_data = torch.load(eval_path, map_location="cpu")
    checkpoint = torch.load(args.resume, map_location="cpu")
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    metrics, ap_per_class = summarize_eval(eval_data)
    result = {
        "exp_name": args.exp_name,
        "seed": args.seed,
        "checkpoint": str(args.resume),
        **metrics,
        "model_size_mb": state_dict_size_mb(state),
        "model_size_protocol": "pure model/EMA state_dict tensor bytes, excludes optimizer and training state",
        "ap_per_class": ap_per_class,
    }

    write_json(output_dir / "results.json", result)
    write_results_csv(output_dir / "results.csv", result)
    write_per_class_csv(output_dir / "per_class_ap.csv", ap_per_class)

    if args.measure_speed:
        speed = measure_speed(args.config, args.resume, args.device, args.input_size, args.warmup, args.repeat)
        speed["seed"] = args.seed
        write_json(output_dir / "speed.json", speed)


if __name__ == "__main__":
    main()
