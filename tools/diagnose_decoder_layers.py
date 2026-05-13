"""Diagnose encoder top-k and decoder-layer refinement without training."""

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

from tools.diagnose_detection_errors import box_cxcywh_to_xyxy, box_iou, summarize_score_iou, target_box_size_wh


def _empty_layer_stats(num_classes, thresholds):
    return {
        "images": 0,
        "total_gt": 0,
        "best_iou_sum": 0.0,
        "hits": {thr: 0 for thr in thresholds},
        "score_iou": [],
        "per_class": [
            {"gt": 0, "best_iou_sum": 0.0, "hits": {thr: 0 for thr in thresholds}}
            for _ in range(num_classes)
        ],
    }


def compute_layer_image_stats(
    pred_boxes_cxcywh,
    pred_logits,
    gt_boxes_xyxy,
    gt_labels,
    thresholds=(0.5, 0.75, 0.9),
    num_classes=80,
    topk=300,
):
    """For one image, measure whether same-class queries cover each GT."""
    thresholds = tuple(float(thr) for thr in thresholds)
    stats = _empty_layer_stats(num_classes, thresholds)
    stats["total_gt"] = int(gt_boxes_xyxy.shape[0])

    for cls_id in gt_labels.tolist():
        cls_id = int(cls_id)
        if 0 <= cls_id < num_classes:
            stats["per_class"][cls_id]["gt"] += 1

    if pred_boxes_cxcywh.numel() == 0 or pred_logits.numel() == 0:
        return stats

    scores, pred_labels = pred_logits.sigmoid().max(dim=-1)
    topk = min(int(topk), int(scores.numel())) if topk > 0 else int(scores.numel())
    order = torch.argsort(scores, descending=True)[:topk]
    pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes_cxcywh[order]).clamp(0, 1)
    pred_scores = scores[order]
    pred_labels = pred_labels[order]

    ious = box_iou(pred_boxes_xyxy, gt_boxes_xyxy)
    best_by_gt = torch.zeros(gt_boxes_xyxy.shape[0], device=gt_boxes_xyxy.device)
    for gt_idx, gt_label in enumerate(gt_labels.tolist()):
        same_class = pred_labels == int(gt_label)
        if same_class.any():
            best_by_gt[gt_idx] = ious[same_class, gt_idx].max()

    stats["best_iou_sum"] = float(best_by_gt.sum().item())
    for thr in thresholds:
        stats["hits"][thr] = int((best_by_gt >= thr).sum().item())

    for gt_idx, gt_label in enumerate(gt_labels.tolist()):
        cls_id = int(gt_label)
        if 0 <= cls_id < num_classes:
            best_iou = float(best_by_gt[gt_idx].item())
            stats["per_class"][cls_id]["best_iou_sum"] += best_iou
            for thr in thresholds:
                if best_iou >= thr:
                    stats["per_class"][cls_id]["hits"][thr] += 1

    best_by_query = torch.zeros(pred_boxes_xyxy.shape[0], device=pred_boxes_xyxy.device)
    if gt_boxes_xyxy.numel() > 0:
        for pred_idx, pred_label in enumerate(pred_labels.tolist()):
            same_gt = gt_labels == int(pred_label)
            if same_gt.any():
                best_by_query[pred_idx] = ious[pred_idx, same_gt].max()
    stats["score_iou"] = [
        [float(score), float(iou)] for score, iou in zip(pred_scores.detach().cpu(), best_by_query.detach().cpu())
    ]
    return stats


def merge_layer_stats(total, image_stats, num_classes, thresholds):
    total["images"] += 1
    total["total_gt"] += image_stats["total_gt"]
    total["best_iou_sum"] += image_stats["best_iou_sum"]
    total.setdefault("score_iou", []).extend(image_stats.get("score_iou", []))
    for thr in thresholds:
        total["hits"][thr] += image_stats["hits"][thr]
    for cls_id in range(num_classes):
        dst = total["per_class"][cls_id]
        src = image_stats["per_class"][cls_id]
        dst["gt"] += src["gt"]
        dst["best_iou_sum"] += src["best_iou_sum"]
        for thr in thresholds:
            dst["hits"][thr] += src["hits"][thr]


def finalize_layer_stats(stats, thresholds, class_names=None):
    total_gt = max(stats["total_gt"], 1)
    result = {
        "images": stats["images"],
        "total_gt": stats["total_gt"],
        "mean_best_iou": stats["best_iou_sum"] / total_gt,
        "recall": {str(thr): stats["hits"][thr] / total_gt for thr in thresholds},
        "hits": {str(thr): stats["hits"][thr] for thr in thresholds},
        "score_iou": summarize_score_iou(torch.tensor(stats.get("score_iou", []), dtype=torch.float32)),
        "per_class": [],
    }

    for cls_id, cls_stats in enumerate(stats["per_class"]):
        cls_gt = max(cls_stats["gt"], 1)
        name = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)
        result["per_class"].append(
            {
                "class_id": cls_id,
                "class_name": name,
                "total_gt": cls_stats["gt"],
                "mean_best_iou": cls_stats["best_iou_sum"] / cls_gt,
                "recall": {str(thr): cls_stats["hits"][thr] / cls_gt for thr in thresholds},
                "hits": {str(thr): cls_stats["hits"][thr] for thr in thresholds},
            }
        )
    return result


def normalized_gt_boxes(target, device):
    raw_boxes = target["boxes"]
    boxes = raw_boxes.to(device=device, dtype=torch.float32)
    size_wh = target_box_size_wh(target, device)
    scale = torch.stack([size_wh[0], size_wh[1], size_wh[0], size_wh[1]])
    return (boxes / scale).clamp(0, 1)


def decoder_layer_outputs(model, images):
    backbone_feats = model.backbone(images)
    encoder_feats = model.encoder(backbone_feats)
    transformer = model.decoder
    memory, spatial_shapes, level_start_index = transformer._get_encoder_input(encoder_feats)
    target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = transformer._get_decoder_input(
        memory, spatial_shapes, None, None
    )

    was_training = transformer.decoder.training
    transformer.decoder.training = True
    try:
        out_bboxes, out_logits = transformer.decoder(
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            transformer.dec_bbox_head,
            transformer.dec_score_head,
            transformer.query_pos_head,
            attn_mask=None,
        )
    finally:
        transformer.decoder.training = was_training

    layers = [("encoder_topk", enc_topk_bboxes, enc_topk_logits)]
    for idx in range(out_bboxes.shape[0]):
        layers.append((f"decoder_{idx}", out_bboxes[idx], out_logits[idx]))
    return layers


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


def print_summary(final_layers, thresholds):
    print("Decoder-layer same-class GT coverage")
    header = ["layer", "meanIoU", *[f"R@{thr:g}" for thr in thresholds], "highScoreLowIoU"]
    print("  ".join(f"{x:>16}" for x in header))
    for item in final_layers:
        row = [
            item["name"],
            f"{item['mean_best_iou']:.4f}",
            *[f"{item['recall'][str(thr)]:.4f}" for thr in thresholds],
            str(item["score_iou"]["high_score_low_iou"]),
        ]
        print("  ".join(f"{x:>16}" for x in row))

    key_thr = "0.9" if 0.9 in thresholds else str(thresholds[-1])
    print(f"\nPer-class R@{key_thr}:")
    for item in final_layers:
        print(f"  {item['name']}")
        for cls_stats in item["per_class"]:
            if cls_stats["total_gt"] > 0:
                print(f"    {cls_stats['class_name']}: {cls_stats['recall'][key_thr]:.4f}")


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
    totals = {}
    image_count = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break

            images = images.to(args.device)
            layers = decoder_layer_outputs(model, images)
            for name, _, _ in layers:
                totals.setdefault(name, _empty_layer_stats(num_classes, thresholds))

            for image_idx, target in enumerate(targets):
                gt_boxes = normalized_gt_boxes(target, args.device)
                gt_labels = target["labels"].to(device=args.device, dtype=torch.int64)
                for name, boxes, logits in layers:
                    image_stats = compute_layer_image_stats(
                        boxes[image_idx],
                        logits[image_idx],
                        gt_boxes,
                        gt_labels,
                        thresholds=thresholds,
                        num_classes=num_classes,
                        topk=args.topk,
                    )
                    merge_layer_stats(totals[name], image_stats, num_classes, thresholds)
                image_count += 1

            if args.print_freq > 0 and (batch_idx + 1) % args.print_freq == 0:
                print(f"processed_batches={batch_idx + 1} images={image_count}")

    final_layers = []
    for name, stats in totals.items():
        item = finalize_layer_stats(stats, thresholds, class_names)
        item["name"] = name
        final_layers.append(item)

    meta = {
        "config": os.fspath(args.config),
        "checkpoint": os.fspath(args.resume),
        "topk": args.topk,
        "images": image_count,
        "layers": final_layers,
    }
    print_summary(final_layers, thresholds)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote JSON: {args.output_json}")


if __name__ == "__main__":
    main()
