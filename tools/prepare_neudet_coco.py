"""Convert the official NEU-DET VOC-style annotations to COCO format."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_src = repo_root.parent / "NEU-DET" / "NEU-DET"
    default_out = repo_root.parent / "NEU-DET-COCO-721"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, default=default_src)
    parser.add_argument("--out-root", type=Path, default=default_out)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def image_group(xml_path: Path) -> str:
    stem = xml_path.stem
    name, _, idx = stem.rpartition("_")
    if name in CLASS_TO_ID and idx.isdigit():
        return name
    raise ValueError(f"Cannot infer NEU-DET class from file name: {xml_path.name}")


def collect_splits(xml_paths: list[Path], seed: int, train_ratio: float, val_ratio: float) -> dict[str, list[Path]]:
    by_group: dict[str, list[Path]] = defaultdict(list)
    for xml_path in xml_paths:
        by_group[image_group(xml_path)].append(xml_path)

    missing = sorted(set(CLASSES) - set(by_group))
    if missing:
        raise ValueError(f"Missing classes: {missing}")

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for class_name in CLASSES:
        items = sorted(by_group[class_name], key=lambda p: p.name)
        rng.shuffle(items)
        train_count = int(len(items) * train_ratio)
        val_count = int(len(items) * val_ratio)
        splits["train"].extend(items[:train_count])
        splits["val"].extend(items[train_count : train_count + val_count])
        splits["test"].extend(items[train_count + val_count :])

    for split_name in splits:
        splits[split_name].sort(key=lambda p: (image_group(p), p.name))
    return splits


def parse_annotation(xml_path: Path, images_dir: Path, image_id: int, ann_start_id: int) -> tuple[dict, list[dict]]:
    root = ET.parse(xml_path).getroot()
    raw_filename = root.findtext("filename") or f"{xml_path.stem}.jpg"
    candidates = [Path(raw_filename)]
    if candidates[0].suffix == "":
        candidates.append(candidates[0].with_suffix(".jpg"))
    candidates.append(Path(f"{xml_path.stem}.jpg"))

    image_path = None
    for candidate in candidates:
        candidate_path = images_dir / candidate.name
        if candidate_path.exists():
            image_path = candidate_path
            break
    if image_path is None:
        raise FileNotFoundError(f"Image not found for {xml_path.name}: {images_dir / raw_filename}")

    filename = image_path.name

    width = int(root.findtext("size/width"))
    height = int(root.findtext("size/height"))
    image = {
        "id": image_id,
        "file_name": filename,
        "width": width,
        "height": height,
    }

    annotations = []
    ann_id = ann_start_id
    for obj in root.findall("object"):
        class_name = obj.findtext("name")
        if class_name not in CLASS_TO_ID:
            raise ValueError(f"Unknown class {class_name!r} in {xml_path.name}")

        xmin = int(float(obj.findtext("bndbox/xmin")))
        ymin = int(float(obj.findtext("bndbox/ymin")))
        xmax = int(float(obj.findtext("bndbox/xmax")))
        ymax = int(float(obj.findtext("bndbox/ymax")))

        xmin = max(0, min(xmin, width))
        ymin = max(0, min(ymin, height))
        xmax = max(0, min(xmax, width))
        ymax = max(0, min(ymax, height))

        box_width = xmax - xmin
        box_height = ymax - ymin
        if box_width <= 0 or box_height <= 0:
            raise ValueError(f"Invalid box in {xml_path.name}: {(xmin, ymin, xmax, ymax)}")

        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": CLASS_TO_ID[class_name],
                "bbox": [xmin, ymin, box_width, box_height],
                "area": box_width * box_height,
                "iscrowd": 0,
            }
        )
        ann_id += 1

    return image, annotations


def build_coco(split_xmls: list[Path], images_dir: Path, target_dir: Path, split_name: str) -> dict:
    split_image_dir = target_dir / f"{split_name}2017"
    split_image_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    ann_id = 1
    for image_id, xml_path in enumerate(split_xmls, start=1):
        image, anns = parse_annotation(xml_path, images_dir, image_id, ann_id)
        shutil.copy2(images_dir / image["file_name"], split_image_dir / image["file_name"])
        images.append(image)
        annotations.extend(anns)
        ann_id += len(anns)

    return {
        "info": {"description": f"NEU-DET {split_name} split", "version": "1.0"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": idx, "name": name, "supercategory": "defect"} for idx, name in enumerate(CLASSES)],
    }


def prepare_output(out_root: Path, overwrite: bool) -> None:
    out_root = out_root.resolve()
    if out_root.exists():
        if not overwrite:
            raise FileExistsError(f"{out_root} already exists. Pass --overwrite to replace it.")
        if len(out_root.parts) < 3:
            raise ValueError(f"Refusing to remove suspicious path: {out_root}")
        shutil.rmtree(out_root)
    (out_root / "annotations").mkdir(parents=True, exist_ok=True)


def print_stats(split_name: str, coco: dict) -> None:
    image_groups = Counter(Path(image["file_name"]).stem.rpartition("_")[0] for image in coco["images"])
    ann_groups = Counter(CLASSES[ann["category_id"]] for ann in coco["annotations"])
    print(f"{split_name}: images={len(coco['images'])}, annotations={len(coco['annotations'])}")
    print("  image_groups:", dict(sorted(image_groups.items())))
    print("  annotation_classes:", dict(sorted(ann_groups.items())))


def main() -> None:
    args = parse_args()
    src_root = args.src_root.resolve()
    out_root = args.out_root.resolve()
    annotations_dir = src_root / "ANNOTATIONS"
    images_dir = src_root / "IMAGES"

    if not annotations_dir.is_dir() or not images_dir.is_dir():
        raise FileNotFoundError(f"Expected ANNOTATIONS and IMAGES under {src_root}")

    xml_paths = sorted(annotations_dir.glob("*.xml"))
    if not xml_paths:
        raise FileNotFoundError(f"No xml files found under {annotations_dir}")

    prepare_output(out_root, args.overwrite)
    splits = collect_splits(xml_paths, args.split_seed, args.train_ratio, args.val_ratio)

    for split_name, split_xmls in splits.items():
        coco = build_coco(split_xmls, images_dir, out_root, split_name)
        json_path = out_root / "annotations" / f"instances_{split_name}2017.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False)
        print_stats(split_name, coco)

    print(f"Done: {out_root}")


if __name__ == "__main__":
    main()
