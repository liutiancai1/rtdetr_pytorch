"""Diagnose final RT-DETR detection errors without training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FP_TYPES = ("duplicate", "class_confusion", "localization", "background")
IOU_BINS = (
    ("0-0.3", 0.0, 0.3),
    ("0.3-0.5", 0.3, 0.5),
    ("0.5-0.75", 0.5, 0.75),
    ("0.75-0.9", 0.75, 0.9),
    ("0.9-1.0", 0.9, 1.01),
)


def box_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-12)


def box_cxcywh_to_xyxy(boxes):
    x_c, y_c, w, h = boxes.unbind(-1)
    return torch.stack([x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], dim=-1)


def _empty_fp():
    return {name: 0 for name in FP_TYPES}


def _empty_image_stats(num_classes):
    return {
        "gt": 0,
        "pred": 0,
        "matched_gt": 0,
        "missed_gt": 0,
        "fp": _empty_fp(),
        "best_iou_bins": {name: 0 for name, _, _ in IOU_BINS},
        "score_iou": [],
        "per_class": [
            {
                "gt": 0,
                "pred": 0,
                "matched_gt": 0,
                "missed_gt": 0,
                "fp": _empty_fp(),
            }
            for _ in range(num_classes)
        ],
    }


def _classify_fp(pred_idx, pred_label, ious, gt_labels, matched_gt, iou_threshold, low_iou):
    if ious.numel() == 0:
        return "background"

    same_class = gt_labels == pred_label
    same_ious = ious[pred_idx, same_class]
    same_indices = torch.where(same_class)[0]
    if same_ious.numel() > 0:
        best_same_pos = int(torch.argmax(same_ious).item())
        best_same_iou = float(same_ious[best_same_pos].item())
        best_same_gt = int(same_indices[best_same_pos].item())
        if best_same_iou >= iou_threshold and best_same_gt in matched_gt:
            return "duplicate"
        if best_same_iou >= low_iou:
            return "localization"

    diff_class = gt_labels != pred_label
    diff_ious = ious[pred_idx, diff_class]
    if diff_ious.numel() > 0 and float(diff_ious.max().item()) >= iou_threshold:
        return "class_confusion"

    if float(ious[pred_idx].max().item()) >= low_iou:
        return "localization"
    return "background"


def diagnose_image_errors(
    pred_boxes,
    pred_scores,
    pred_labels,
    gt_boxes,
    gt_labels,
    iou_threshold=0.5,
    low_iou=0.1,
    num_classes=80,
    max_dets=100,
):
    """Greedy final-prediction error split for one image and one IoU threshold."""
    stats = _empty_image_stats(num_classes)
    order = torch.argsort(pred_scores, descending=True)[:max_dets]
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]
    pred_labels = pred_labels[order]

    stats["gt"] = int(gt_boxes.shape[0])
    stats["pred"] = int(pred_boxes.shape[0])
    for cls_id in gt_labels.tolist():
        if 0 <= int(cls_id) < num_classes:
            stats["per_class"][int(cls_id)]["gt"] += 1
    for cls_id in pred_labels.tolist():
        if 0 <= int(cls_id) < num_classes:
            stats["per_class"][int(cls_id)]["pred"] += 1

    ious = box_iou(pred_boxes, gt_boxes)
    matched_gt = set()
    for pred_idx, pred_label in enumerate(pred_labels.tolist()):
        same_class = gt_labels == int(pred_label)
        candidate_gt = torch.where(same_class)[0]
        if candidate_gt.numel() == 0:
            fp_type = _classify_fp(pred_idx, int(pred_label), ious, gt_labels, matched_gt, iou_threshold, low_iou)
            stats["fp"][fp_type] += 1
            if 0 <= int(pred_label) < num_classes:
                stats["per_class"][int(pred_label)]["fp"][fp_type] += 1
            continue

        same_ious = ious[pred_idx, candidate_gt]
        best_pos = int(torch.argmax(same_ious).item())
        best_iou = float(same_ious[best_pos].item())
        best_gt = int(candidate_gt[best_pos].item())
        if best_iou >= iou_threshold and best_gt not in matched_gt:
            matched_gt.add(best_gt)
            stats["matched_gt"] += 1
            gt_cls = int(gt_labels[best_gt].item())
            if 0 <= gt_cls < num_classes:
                stats["per_class"][gt_cls]["matched_gt"] += 1
        else:
            fp_type = _classify_fp(pred_idx, int(pred_label), ious, gt_labels, matched_gt, iou_threshold, low_iou)
            stats["fp"][fp_type] += 1
            if 0 <= int(pred_label) < num_classes:
                stats["per_class"][int(pred_label)]["fp"][fp_type] += 1

    stats["missed_gt"] = stats["gt"] - stats["matched_gt"]
    for cls_id in range(num_classes):
        stats["per_class"][cls_id]["missed_gt"] = (
            stats["per_class"][cls_id]["gt"] - stats["per_class"][cls_id]["matched_gt"]
        )

    if gt_boxes.numel() > 0:
        best_same_by_gt = torch.zeros(gt_boxes.shape[0], device=gt_boxes.device)
        for gt_idx, gt_label in enumerate(gt_labels.tolist()):
            same_preds = pred_labels == int(gt_label)
            if same_preds.any():
                best_same_by_gt[gt_idx] = ious[same_preds, gt_idx].max()
        for name, low, high in IOU_BINS:
            stats["best_iou_bins"][name] = int(((best_same_by_gt >= low) & (best_same_by_gt < high)).sum().item())

    if pred_boxes.numel() > 0:
        best_same_by_pred = torch.zeros(pred_boxes.shape[0], device=pred_boxes.device)
        for pred_idx, pred_label in enumerate(pred_labels.tolist()):
            same_gt = gt_labels == int(pred_label)
            if same_gt.any():
                best_same_by_pred[pred_idx] = ious[pred_idx, same_gt].max()
        stats["score_iou"] = [
            [float(score), float(iou)] for score, iou in zip(pred_scores.detach().cpu(), best_same_by_pred.detach().cpu())
        ]

    return stats


def merge_error_stats(total, image_stats, num_classes):
    total["gt"] += image_stats["gt"]
    total["pred"] += image_stats["pred"]
    total["matched_gt"] += image_stats["matched_gt"]
    total["missed_gt"] += image_stats["missed_gt"]
    total["score_iou"].extend(image_stats["score_iou"])
    for key in FP_TYPES:
        total["fp"][key] += image_stats["fp"][key]
    for key, _, _ in IOU_BINS:
        total["best_iou_bins"][key] += image_stats["best_iou_bins"][key]
    for cls_id in range(num_classes):
        dst = total["per_class"][cls_id]
        src = image_stats["per_class"][cls_id]
        dst["gt"] += src["gt"]
        dst["pred"] += src["pred"]
        dst["matched_gt"] += src["matched_gt"]
        dst["missed_gt"] += src["missed_gt"]
        for key in FP_TYPES:
            dst["fp"][key] += src["fp"][key]


def _empty_total(num_classes):
    stats = _empty_image_stats(num_classes)
    stats["images"] = 0
    return stats


def finalize_error_stats(stats, class_names=None):
    total_gt = max(stats["gt"], 1)
    fp_total = sum(stats["fp"].values())
    score_iou = torch.tensor(stats["score_iou"], dtype=torch.float32) if stats["score_iou"] else torch.zeros((0, 2))
    result = {
        "images": stats["images"],
        "gt": stats["gt"],
        "pred": stats["pred"],
        "matched_gt": stats["matched_gt"],
        "missed_gt": stats["missed_gt"],
        "recall": stats["matched_gt"] / total_gt,
        "fp": {key: stats["fp"][key] for key in FP_TYPES},
        "fp_total": fp_total,
        "best_iou_bins": stats["best_iou_bins"],
        "score_iou": summarize_score_iou(score_iou),
        "per_class": [],
    }
    for cls_id, cls_stats in enumerate(stats["per_class"]):
        name = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)
        cls_gt = max(cls_stats["gt"], 1)
        result["per_class"].append(
            {
                "class_id": cls_id,
                "class_name": name,
                "gt": cls_stats["gt"],
                "pred": cls_stats["pred"],
                "matched_gt": cls_stats["matched_gt"],
                "missed_gt": cls_stats["missed_gt"],
                "recall": cls_stats["matched_gt"] / cls_gt,
                "fp": cls_stats["fp"],
            }
        )
    return result


def summarize_score_iou(score_iou):
    if score_iou.numel() == 0:
        return {"count": 0, "mean_iou": 0.0, "high_score_low_iou": 0}
    scores = score_iou[:, 0]
    ious = score_iou[:, 1]
    return {
        "count": int(score_iou.shape[0]),
        "mean_iou": float(ious.mean().item()),
        "high_score_low_iou": int(((scores >= 0.5) & (ious < 0.5)).sum().item()),
        "score_ge_0.5": int((scores >= 0.5).sum().item()),
    }


def target_box_size_wh(target, device):
    boxes = target["boxes"]
    spatial_size = getattr(boxes, "spatial_size", None)
    if spatial_size is not None:
        h, w = spatial_size
        return torch.tensor([w, h], device=device, dtype=torch.float32)
    return target["size"].to(device=device, dtype=torch.float32)


def iter_image_detections(detections):
    if (
        isinstance(detections, (tuple, list))
        and len(detections) == 3
        and all(torch.is_tensor(item) for item in detections)
    ):
        labels, boxes, scores = detections
        for lab, box, score in zip(labels, boxes, scores):
            yield {"labels": lab, "boxes": box, "scores": score}
        return

    yield from detections


def xyxy_from_target(target, device, original_scale=True):
    boxes = target["boxes"].to(device=device, dtype=torch.float32)
    if not original_scale:
        return boxes
    size = target_box_size_wh(target, device)
    orig_size = target["orig_size"].to(device=device, dtype=torch.float32)
    scale = torch.stack([orig_size[0] / size[0], orig_size[1] / size[1], orig_size[0] / size[0], orig_size[1] / size[1]])
    return boxes * scale


def load_model_and_loader(args):
    from src.core import YAMLConfig

    cfg = YAMLConfig(
        str(args.config),
        PResNet={"pretrained": False},
        val_dataloader={"batch_size": args.batch_size, "num_workers": args.num_workers},
    )
    checkpoint = torch.load(args.resume, map_location="cpu")
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)
    model = cfg.model.to(args.device).eval()
    postprocessor = cfg.postprocessor.deploy()
    return cfg, model, postprocessor, cfg.val_dataloader


def get_class_names(data_loader, num_classes):
    names = [str(i) for i in range(num_classes)]
    cats = getattr(getattr(data_loader.dataset, "coco", None), "cats", {})
    for cat_id, cat in cats.items():
        if 0 <= int(cat_id) < num_classes:
            names[int(cat_id)] = cat.get("name", str(cat_id))
    return names


def print_summary(final_by_thr):
    for thr, stats in final_by_thr.items():
        print(f"\nIoU={thr}")
        print(f"  GT={stats['gt']} matched={stats['matched_gt']} recall={stats['recall']:.4f}")
        print(f"  FP={stats['fp_total']} {stats['fp']}")
        print(f"  best_iou_bins={stats['best_iou_bins']}")
        print(f"  score_iou={stats['score_iou']}")
        print("  per-class recall:")
        for item in stats["per_class"]:
            if item["gt"] > 0:
                print(f"    {item['class_name']}: recall={item['recall']:.4f} missed={item['missed_gt']} fp={item['fp']}")


def parse_args():
    default_resume = REPO_ROOT / "output" / "experiments" / "rtdetr_r18_neudet_baseline_300e" / "seed_42" / "best.pth"
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "rtdetr" / "rtdetr_r18vd_6x_neudet.yml")
    parser.add_argument("--resume", type=Path, default=default_resume)
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-dets", type=int, default=100)
    parser.add_argument("--low-iou", type=float, default=0.1)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.5, 0.75, 0.9])
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--print-freq", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    args.device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if not args.resume.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.resume}")

    cfg, model, postprocessor, data_loader = load_model_and_loader(args)
    num_classes = int(cfg.yaml_cfg["num_classes"])
    class_names = get_class_names(data_loader, num_classes)
    totals = {str(thr): _empty_total(num_classes) for thr in args.iou_thresholds}

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            images = images.to(args.device)
            outputs = model(images)
            orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(args.device)
            detections = postprocessor(outputs, orig_sizes)

            for det, target in zip(iter_image_detections(detections), targets):
                gt_boxes = xyxy_from_target(target, args.device, original_scale=True)
                gt_labels = target["labels"].to(device=args.device, dtype=torch.int64)
                for thr in args.iou_thresholds:
                    key = str(thr)
                    image_stats = diagnose_image_errors(
                        det["boxes"].to(args.device),
                        det["scores"].to(args.device),
                        det["labels"].to(args.device),
                        gt_boxes,
                        gt_labels,
                        iou_threshold=float(thr),
                        low_iou=args.low_iou,
                        num_classes=num_classes,
                        max_dets=args.max_dets,
                    )
                    merge_error_stats(totals[key], image_stats, num_classes)
                    totals[key]["images"] += 1

            if args.print_freq > 0 and (batch_idx + 1) % args.print_freq == 0:
                print(f"processed_batches={batch_idx + 1}")

    final = {key: finalize_error_stats(stats, class_names) for key, stats in totals.items()}
    meta = {
        "config": os.fspath(args.config),
        "checkpoint": os.fspath(args.resume),
        "max_dets": args.max_dets,
        "low_iou": args.low_iou,
        "diagnostics": final,
    }
    print_summary(final)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote JSON: {args.output_json}")


if __name__ == "__main__":
    main()
