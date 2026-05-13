"""Diagnose RT-DETR encoder query selection without training."""

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

from tools.diagnose_detection_errors import target_box_size_wh


METHODS = (
    "class_topk",
    "candidate_pool",
    "oracle_candidate_to_topk",
    "oracle_all_to_topk",
)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    return torch.stack(
        [
            x_c - 0.5 * w,
            y_c - 0.5 * h,
            x_c + 0.5 * w,
            y_c + 0.5 * h,
        ],
        dim=-1,
    )


def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-12), union


def _empty_method_stats(iou_thresholds):
    return {
        "hits": {thr: 0 for thr in iou_thresholds},
        "best_iou_sum": 0.0,
    }


def _empty_stats(iou_thresholds, num_classes):
    return {
        "total_gt": 0,
        "methods": {name: _empty_method_stats(iou_thresholds) for name in METHODS},
        "per_class": [
            {
                "total_gt": 0,
                "methods": {name: _empty_method_stats(iou_thresholds) for name in METHODS},
            }
            for _ in range(num_classes)
        ],
    }


def _add_method_stats(dst, best_ious, iou_thresholds):
    dst["best_iou_sum"] += float(best_ious.sum().item())
    for thr in iou_thresholds:
        dst["hits"][thr] += int((best_ious >= thr).sum().item())


def _topk_indices(scores, k):
    k = min(k, scores.numel())
    if k <= 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    return torch.topk(scores, k, dim=0).indices


def _selected_best_ious(iou_matrix, selected_indices):
    if iou_matrix.numel() == 0:
        return torch.zeros(0, device=iou_matrix.device)
    if selected_indices.numel() == 0:
        return torch.zeros(iou_matrix.shape[1], device=iou_matrix.device)
    return iou_matrix[selected_indices].max(dim=0).values


def compute_image_stats(
    proposals_cxcywh,
    class_logits,
    gt_boxes_xyxy,
    gt_labels,
    topk=300,
    candidate_topk=600,
    iou_thresholds=(0.5, 0.75, 0.9),
    num_classes=80,
):
    """Return recall-oriented query selection stats for one image."""
    stats = _empty_stats(iou_thresholds, num_classes)
    num_gt = int(gt_boxes_xyxy.shape[0])
    stats["total_gt"] = num_gt
    for cls_id in gt_labels.tolist():
        if 0 <= int(cls_id) < num_classes:
            stats["per_class"][int(cls_id)]["total_gt"] += 1

    if num_gt == 0 or proposals_cxcywh.numel() == 0:
        return stats

    topk = min(topk, proposals_cxcywh.shape[0])
    candidate_topk = min(max(candidate_topk, topk), proposals_cxcywh.shape[0])

    proposal_boxes = box_cxcywh_to_xyxy(proposals_cxcywh).clamp(0, 1)
    iou_matrix, _ = box_iou(proposal_boxes, gt_boxes_xyxy.clamp(0, 1))
    oracle_scores = iou_matrix.max(dim=1).values
    class_scores = class_logits.max(dim=-1).values

    class_topk = _topk_indices(class_scores, topk)
    candidate_pool = _topk_indices(class_scores, candidate_topk)
    oracle_candidate = candidate_pool[_topk_indices(oracle_scores[candidate_pool], topk)]
    oracle_all = _topk_indices(oracle_scores, topk)

    selections = {
        "class_topk": class_topk,
        "candidate_pool": candidate_pool,
        "oracle_candidate_to_topk": oracle_candidate,
        "oracle_all_to_topk": oracle_all,
    }

    for method, selected in selections.items():
        best_ious = _selected_best_ious(iou_matrix, selected)
        _add_method_stats(stats["methods"][method], best_ious, iou_thresholds)

        for cls_id in range(num_classes):
            mask = gt_labels == cls_id
            if mask.any():
                _add_method_stats(
                    stats["per_class"][cls_id]["methods"][method],
                    best_ious[mask],
                    iou_thresholds,
                )

    return stats


def merge_stats(total, image_stats, iou_thresholds, num_classes):
    total["total_gt"] += image_stats["total_gt"]
    for method in METHODS:
        total["methods"][method]["best_iou_sum"] += image_stats["methods"][method]["best_iou_sum"]
        for thr in iou_thresholds:
            total["methods"][method]["hits"][thr] += image_stats["methods"][method]["hits"][thr]

    for cls_id in range(num_classes):
        total["per_class"][cls_id]["total_gt"] += image_stats["per_class"][cls_id]["total_gt"]
        for method in METHODS:
            total["per_class"][cls_id]["methods"][method]["best_iou_sum"] += (
                image_stats["per_class"][cls_id]["methods"][method]["best_iou_sum"]
            )
            for thr in iou_thresholds:
                total["per_class"][cls_id]["methods"][method]["hits"][thr] += (
                    image_stats["per_class"][cls_id]["methods"][method]["hits"][thr]
                )


def finalize_stats(stats, iou_thresholds, class_names=None):
    result = {
        "total_gt": stats["total_gt"],
        "methods": {},
        "per_class": [],
    }

    for method in METHODS:
        result["methods"][method] = _finalize_method(stats["methods"][method], stats["total_gt"], iou_thresholds)

    for cls_id, cls_stats in enumerate(stats["per_class"]):
        name = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)
        result["per_class"].append(
            {
                "class_id": cls_id,
                "class_name": name,
                "total_gt": cls_stats["total_gt"],
                "methods": {
                    method: _finalize_method(cls_stats["methods"][method], cls_stats["total_gt"], iou_thresholds)
                    for method in METHODS
                },
            }
        )
    return result


def _finalize_method(method_stats, total_gt, iou_thresholds):
    if total_gt <= 0:
        return {
            "mean_best_iou": 0.0,
            "recall": {str(thr): 0.0 for thr in iou_thresholds},
            "hits": {str(thr): 0 for thr in iou_thresholds},
        }
    return {
        "mean_best_iou": method_stats["best_iou_sum"] / total_gt,
        "recall": {str(thr): method_stats["hits"][thr] / total_gt for thr in iou_thresholds},
        "hits": {str(thr): method_stats["hits"][thr] for thr in iou_thresholds},
    }


def encoder_proposals(model, images):
    backbone_feats = model.backbone(images)
    encoder_feats = model.encoder(backbone_feats)
    decoder = model.decoder
    memory, spatial_shapes, _ = decoder._get_encoder_input(encoder_feats)

    if decoder.training or decoder.eval_spatial_size is None:
        anchors, valid_mask = decoder._generate_anchors(spatial_shapes, device=memory.device)
    else:
        anchors = decoder.anchors.to(memory.device)
        valid_mask = decoder.valid_mask.to(memory.device)

    memory = valid_mask.to(memory.dtype) * memory
    output_memory = decoder.enc_output(memory)
    class_logits = decoder.enc_score_head(output_memory)
    boxes = torch.sigmoid(decoder.enc_bbox_head(output_memory) + anchors)
    return boxes, class_logits


def normalized_gt_boxes(target, device):
    raw_boxes = target["boxes"]
    boxes = raw_boxes.to(device=device, dtype=torch.float32)
    size_wh = target_box_size_wh(target, device)
    scale = torch.stack([size_wh[0], size_wh[1], size_wh[0], size_wh[1]])
    return (boxes / scale).clamp(0, 1)


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
    return cfg, model, cfg.val_dataloader


def get_class_names(data_loader, num_classes):
    names = [str(i) for i in range(num_classes)]
    cats = getattr(getattr(data_loader.dataset, "coco", None), "cats", {})
    for cat_id, cat in cats.items():
        if 0 <= int(cat_id) < num_classes:
            names[int(cat_id)] = cat.get("name", str(cat_id))
    return names


def print_summary(final_stats, iou_thresholds, topk, candidate_topk):
    labels = {
        "class_topk": f"class_top{topk}",
        "candidate_pool": f"class_top{candidate_topk}_pool",
        "oracle_candidate_to_topk": f"oracle_top{candidate_topk}_to_top{topk}",
        "oracle_all_to_topk": f"oracle_all_to_top{topk}",
    }

    print(f"Total GT: {final_stats['total_gt']}")
    header = ["method", "meanIoU", *[f"R@{thr:g}" for thr in iou_thresholds]]
    print("  ".join(f"{x:>22}" for x in header))
    for method in METHODS:
        row = [labels[method], f"{final_stats['methods'][method]['mean_best_iou']:.4f}"]
        row.extend(f"{final_stats['methods'][method]['recall'][str(thr)]:.4f}" for thr in iou_thresholds)
        print("  ".join(f"{x:>22}" for x in row))

    print("\nPer-class R@0.75:")
    print(f"{'class':>22}  {'class_topk':>12}  {'pool':>12}  {'oracle_pool':>12}  {'oracle_all':>12}")
    for item in final_stats["per_class"]:
        if item["total_gt"] == 0:
            continue
        r = item["methods"]
        print(
            f"{item['class_name']:>22}  "
            f"{r['class_topk']['recall']['0.75']:>12.4f}  "
            f"{r['candidate_pool']['recall']['0.75']:>12.4f}  "
            f"{r['oracle_candidate_to_topk']['recall']['0.75']:>12.4f}  "
            f"{r['oracle_all_to_topk']['recall']['0.75']:>12.4f}"
        )


def parse_args():
    default_resume = REPO_ROOT / "output" / "experiments" / "rtdetr_r18_neudet_baseline_300e" / "seed_42" / "best.pth"
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "rtdetr" / "rtdetr_r18vd_6x_neudet.yml")
    parser.add_argument("--resume", type=Path, default=default_resume)
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--topk", type=int, default=300)
    parser.add_argument("--candidate-topk", type=int, default=600)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.5, 0.75, 0.9])
    parser.add_argument("--max-batches", type=int, default=0, help="0 means all validation batches.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--print-freq", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    args.device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if not args.resume.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.resume}")

    cfg, model, data_loader = load_model_and_loader(args)
    num_classes = int(cfg.yaml_cfg["num_classes"])
    class_names = get_class_names(data_loader, num_classes)
    thresholds = tuple(float(x) for x in args.iou_thresholds)
    totals = _empty_stats(thresholds, num_classes)
    image_count = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break

            images = images.to(args.device)
            proposals, logits = encoder_proposals(model, images)
            for image_idx, target in enumerate(targets):
                gt_boxes = normalized_gt_boxes(target, args.device)
                gt_labels = target["labels"].to(device=args.device, dtype=torch.int64)
                image_stats = compute_image_stats(
                    proposals[image_idx],
                    logits[image_idx],
                    gt_boxes,
                    gt_labels,
                    topk=args.topk,
                    candidate_topk=args.candidate_topk,
                    iou_thresholds=thresholds,
                    num_classes=num_classes,
                )
                merge_stats(totals, image_stats, thresholds, num_classes)
                image_count += 1

            if args.print_freq > 0 and (batch_idx + 1) % args.print_freq == 0:
                print(f"processed_batches={batch_idx + 1} images={image_count}")

    final = finalize_stats(totals, thresholds, class_names)
    final["images"] = image_count
    final["topk"] = args.topk
    final["candidate_topk"] = args.candidate_topk
    final["checkpoint"] = os.fspath(args.resume)
    final["config"] = os.fspath(args.config)

    print_summary(final, thresholds, args.topk, args.candidate_topk)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote JSON: {args.output_json}")


if __name__ == "__main__":
    main()
