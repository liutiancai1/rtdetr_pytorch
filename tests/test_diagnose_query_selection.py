import unittest

import torch

from tools.diagnose_query_selection import compute_image_stats


class QuerySelectionDiagnosticsTest(unittest.TestCase):
    def test_oracle_rerank_shows_hidden_good_candidate(self):
        proposals = torch.tensor(
            [
                [0.20, 0.20, 0.20, 0.20],
                [0.50, 0.50, 0.20, 0.20],
                [0.80, 0.80, 0.20, 0.20],
            ],
            dtype=torch.float32,
        )
        logits = torch.tensor(
            [
                [4.0, -4.0],
                [3.0, -4.0],
                [2.0, -4.0],
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

        stats = compute_image_stats(
            proposals,
            logits,
            gt_boxes,
            gt_labels,
            topk=2,
            candidate_topk=3,
            iou_thresholds=(0.5, 0.75),
            num_classes=2,
        )

        self.assertEqual(stats["total_gt"], 2)
        self.assertEqual(stats["methods"]["class_topk"]["hits"][0.5], 1)
        self.assertEqual(stats["methods"]["class_topk"]["hits"][0.75], 1)
        self.assertEqual(stats["methods"]["candidate_pool"]["hits"][0.5], 2)
        self.assertEqual(stats["methods"]["oracle_candidate_to_topk"]["hits"][0.5], 2)
        self.assertEqual(stats["methods"]["oracle_all_to_topk"]["hits"][0.75], 2)
        self.assertEqual(stats["per_class"][1]["methods"]["class_topk"]["hits"][0.5], 0)
        self.assertEqual(stats["per_class"][1]["methods"]["oracle_candidate_to_topk"]["hits"][0.5], 1)


if __name__ == "__main__":
    unittest.main()
