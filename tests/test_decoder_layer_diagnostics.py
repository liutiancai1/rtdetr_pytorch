import unittest

import torch

from tools.diagnose_decoder_layers import compute_layer_image_stats, finalize_layer_stats, merge_layer_stats


class DecoderLayerDiagnosticsTest(unittest.TestCase):
    def test_later_layer_can_show_iou_refinement(self):
        logits = torch.tensor(
            [
                [5.0, -5.0],
                [-5.0, 5.0],
            ],
            dtype=torch.float32,
        )
        gt_boxes = torch.tensor(
            [
                [0.10, 0.10, 0.30, 0.30],
                [0.70, 0.70, 0.90, 0.90],
            ],
            dtype=torch.float32,
        )
        gt_labels = torch.tensor([0, 1], dtype=torch.int64)

        layer0_boxes = torch.tensor(
            [
                [0.20, 0.20, 0.20, 0.20],
                [0.75, 0.75, 0.20, 0.20],
            ],
            dtype=torch.float32,
        )
        layer1_boxes = torch.tensor(
            [
                [0.20, 0.20, 0.20, 0.20],
                [0.80, 0.80, 0.20, 0.20],
            ],
            dtype=torch.float32,
        )

        stats0 = compute_layer_image_stats(layer0_boxes, logits, gt_boxes, gt_labels, (0.5, 0.75), 2, topk=2)
        stats1 = compute_layer_image_stats(layer1_boxes, logits, gt_boxes, gt_labels, (0.5, 0.75), 2, topk=2)

        self.assertEqual(stats0["total_gt"], 2)
        self.assertEqual(stats0["hits"][0.5], 1)
        self.assertEqual(stats0["hits"][0.75], 1)
        self.assertEqual(stats1["hits"][0.75], 2)
        self.assertGreater(stats1["best_iou_sum"], stats0["best_iou_sum"])

    def test_merge_and_finalize_reports_recall(self):
        gt_boxes = torch.tensor([[0.10, 0.10, 0.30, 0.30]], dtype=torch.float32)
        gt_labels = torch.tensor([0], dtype=torch.int64)
        boxes = torch.tensor([[0.20, 0.20, 0.20, 0.20]], dtype=torch.float32)
        logits = torch.tensor([[4.0, -4.0]], dtype=torch.float32)

        image_stats = compute_layer_image_stats(boxes, logits, gt_boxes, gt_labels, (0.5, 0.9), 2, topk=1)
        total = {"images": 0, "total_gt": 0, "best_iou_sum": 0.0, "hits": {0.5: 0, 0.9: 0}, "per_class": []}
        total["per_class"] = [
            {"gt": 0, "best_iou_sum": 0.0, "hits": {0.5: 0, 0.9: 0}},
            {"gt": 0, "best_iou_sum": 0.0, "hits": {0.5: 0, 0.9: 0}},
        ]

        merge_layer_stats(total, image_stats, 2, (0.5, 0.9))
        final = finalize_layer_stats(total, (0.5, 0.9), ["crazing", "inclusion"])

        self.assertEqual(final["total_gt"], 1)
        self.assertEqual(final["recall"]["0.5"], 1.0)
        self.assertEqual(final["recall"]["0.9"], 1.0)
        self.assertEqual(final["per_class"][0]["class_name"], "crazing")
        self.assertEqual(final["per_class"][0]["recall"]["0.5"], 1.0)


if __name__ == "__main__":
    unittest.main()
