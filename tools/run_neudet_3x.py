"""Run a three-seed NEU-DET RT-DETR experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from statistics import mean, stdev
from pathlib import Path


CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
METRIC_FIELDS = ["map50_95", "map50", "map75", "model_size_mb"]
SPEED_FIELDS = ["params_m", "gflops", "fps", "latency_ms"]
TEST_IMG_FOLDER = "../NEU-DET-COCO-721/test2017/"
TEST_ANN_FILE = "../NEU-DET-COCO-721/annotations/instances_test2017.json"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=repo_root / "configs/rtdetr/rtdetr_r18vd_6x_neudet.yml")
    parser.add_argument("--test-config", type=Path, default=None)
    parser.add_argument("--exp-name", default="rtdetr_r18_neudet_baseline_300e")
    parser.add_argument("--output-root", type=Path, default=repo_root / "output/experiments")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 3407])
    parser.add_argument("--speed-seed", type=int, default=42)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--tuning", type=Path, default=None, help="Optional detector checkpoint. Do not set for the backbone-only baseline.")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def as_yaml_path(path: Path) -> str:
    return path.as_posix()


def rel_path(path: Path, start: Path) -> Path:
    return Path(os.path.relpath(path.resolve(), start.resolve()))


def command_text(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in cmd)


def write_wrapper_config(
    path: Path,
    include_path: Path,
    output_dir: Path,
    repo_root: Path,
    use_test_dataset: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    include = as_yaml_path(rel_path(include_path, path.parent))
    try:
        out_dir = as_yaml_path(output_dir.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        out_dir = as_yaml_path(output_dir.resolve())

    lines = [
        "__include__: [",
        f"  '{include}',",
        "]",
        "",
        "",
        f"output_dir: {out_dir}",
        "",
    ]
    if use_test_dataset:
        lines.extend(
            [
                "val_dataloader:",
                "  dataset:",
                f"    img_folder: {TEST_IMG_FOLDER}",
                f"    ann_file: {TEST_ANN_FILE}",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def run(cmd: list[str], log_path: Path, repo_root: Path, dry_run: bool) -> None:
    print(f"$ {command_text(cmd)}")
    if dry_run:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {command_text(cmd)}\n\n")
        log_file.flush()
        subprocess.run(cmd, cwd=repo_root, stdout=log_file, stderr=subprocess.STDOUT, check=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def metric_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def metric_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def write_aggregate(exp_root: Path, exp_name: str, seeds: list[int], speed_seed: int) -> None:
    seed_results = []
    for seed in seeds:
        result_path = exp_root / f"seed_{seed}" / "results.json"
        if result_path.is_file():
            seed_results.append(read_json(result_path))

    if not seed_results:
        return

    mean_metrics = {
        field: metric_mean([float(item[field]) for item in seed_results])
        for field in METRIC_FIELDS
    }
    std_metrics = {
        field: metric_std([float(item[field]) for item in seed_results])
        for field in METRIC_FIELDS
    }

    per_class = {}
    for class_name in CLASSES:
        values = [float(item["ap_per_class"][class_name]) for item in seed_results]
        per_class[class_name] = {
            "mean": metric_mean(values),
            "std": metric_std(values),
        }

    speed_path = exp_root / f"seed_{speed_seed}" / "speed.json"
    speed = read_json(speed_path) if speed_path.is_file() else {}
    root_speed = {field: speed.get(field, 0.0) for field in SPEED_FIELDS}
    if speed:
        root_speed.update(
            {
                "input_size": speed.get("input_size", [640, 640]),
                "batch_size": speed.get("batch_size", 1),
                "warmup": speed.get("warmup", 20),
                "frame_count": speed.get("frame_count", 0),
                "device": speed.get("device", ""),
                "speed_protocol": speed.get("speed_protocol", ""),
                "params_protocol": speed.get("params_protocol", ""),
                "gflops_protocol": speed.get("gflops_protocol", ""),
                "gflops_note": speed.get("gflops_note", ""),
                "gmacs": speed.get("gmacs", 0.0),
                "source_seed": speed_seed,
            }
        )

    aggregate = {
        "exp_name": exp_name,
        "seeds": seeds,
        "mean": mean_metrics,
        "std": std_metrics,
        "speed": root_speed,
        "ap_per_class": per_class,
    }
    (exp_root / "results.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    (exp_root / "speed.json").write_text(json.dumps(root_speed, indent=2, ensure_ascii=False), encoding="utf-8")

    with (exp_root / "results.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["exp_name", "stat", "map50_95", "map50", "map75", *SPEED_FIELDS, "model_size_mb"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        speed_row = {field: root_speed.get(field, 0.0) for field in SPEED_FIELDS}
        writer.writerow({"exp_name": exp_name, "stat": "mean", **mean_metrics, **speed_row})
        writer.writerow({"exp_name": exp_name, "stat": "std", **std_metrics, **{field: 0.0 for field in SPEED_FIELDS}})

    with (exp_root / "per_class_ap.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "mean", "std"])
        writer.writeheader()
        for class_name in CLASSES:
            writer.writerow({"class": class_name, **per_class[class_name]})

    lines = [
        f"Experiment: {exp_name}",
        f"Seeds: {', '.join(str(seed) for seed in seeds)}",
        "",
        "Accuracy:",
        f"  mAP@0.5:0.95: {mean_metrics['map50_95']:.4f} +/- {std_metrics['map50_95']:.4f}",
        f"  mAP@0.5:      {mean_metrics['map50']:.4f} +/- {std_metrics['map50']:.4f}",
        f"  mAP@0.75:     {mean_metrics['map75']:.4f} +/- {std_metrics['map75']:.4f}",
        "",
        "Complexity / Speed:",
        f"  Params:       {root_speed.get('params_m', 0.0):.4f} M",
        f"  GFLOPs:       {root_speed.get('gflops', 0.0):.4f}",
        f"  FPS:          {root_speed.get('fps', 0.0):.4f}",
        f"  Latency:      {root_speed.get('latency_ms', 0.0):.4f} ms/image",
        f"  Model size:   {mean_metrics['model_size_mb']:.4f} +/- {std_metrics['model_size_mb']:.4f} MB",
        "",
        "Per-class AP:",
    ]
    for class_name in CLASSES:
        item = per_class[class_name]
        lines.append(f"  {class_name}: {item['mean']:.4f} +/- {item['std']:.4f}")
    (exp_root / "metrics_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    train_py = repo_root / "tools/train.py"
    metrics_py = repo_root / "tools/export_neudet_metrics.py"
    config = args.config.resolve()
    test_config = args.test_config.resolve() if args.test_config is not None else config
    output_root = args.output_root.resolve()
    exp_root = output_root / args.exp_name

    if not config.is_file():
        raise FileNotFoundError(config)
    if not test_config.is_file():
        raise FileNotFoundError(test_config)

    for seed in args.seeds:
        run_dir = exp_root / f"seed_{seed}"
        cfg_dir = run_dir / "configs"
        train_cfg = cfg_dir / "train.yml"
        test_cfg = cfg_dir / "test.yml"

        if not args.dry_run:
            write_wrapper_config(train_cfg, config, run_dir, repo_root)
            write_wrapper_config(test_cfg, test_config, run_dir, repo_root, use_test_dataset=args.test_config is None)

        train_cmd = [args.python, str(train_py), "-c", str(train_cfg), "--seed", str(seed)]
        if args.tuning is not None:
            train_cmd.extend(["-t", str(args.tuning.resolve())])
        if args.amp:
            train_cmd.append("--amp")

        test_cmd = [
            args.python,
            str(train_py),
            "-c",
            str(test_cfg),
            "-r",
            str(run_dir / "best.pth"),
            "--test-only",
            "--seed",
            str(seed),
        ]
        metrics_cmd = [
            args.python,
            str(metrics_py),
            "--config",
            str(test_cfg),
            "--resume",
            str(run_dir / "best.pth"),
            "--output-dir",
            str(run_dir),
            "--exp-name",
            args.exp_name,
            "--seed",
            str(seed),
        ]
        if seed == args.speed_seed:
            metrics_cmd.append("--measure-speed")

        commands_path = run_dir / "commands.txt"
        if not args.dry_run:
            commands_path.parent.mkdir(parents=True, exist_ok=True)
            commands_path.write_text(
                "\n".join([command_text(train_cmd), command_text(test_cmd), command_text(metrics_cmd), ""]),
                encoding="utf-8",
            )

        if not args.skip_train:
            run(train_cmd, run_dir / "train.log", repo_root, args.dry_run)

        best_path = run_dir / "best.pth"
        if not args.skip_test:
            if not args.dry_run and not best_path.is_file():
                raise FileNotFoundError(f"Missing best checkpoint: {best_path}")
            run(test_cmd, run_dir / "test.log", repo_root, args.dry_run)

        if not args.skip_metrics:
            if not args.dry_run and not (run_dir / "eval.pth").is_file():
                raise FileNotFoundError(f"Missing eval file: {run_dir / 'eval.pth'}")
            run(metrics_cmd, run_dir / "metrics.log", repo_root, args.dry_run)

    if not args.dry_run and not args.skip_metrics:
        write_aggregate(exp_root, args.exp_name, args.seeds, args.speed_seed)


if __name__ == "__main__":
    main()
