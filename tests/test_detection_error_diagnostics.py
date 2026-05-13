import unittest

import torch

from tools.diagnose_detection_errors import diagnose_image_errors, iter_image_detections


class DetectionErrorDiagnosticsTest(unittest.TestCase):
    def test_splits_final_predictions_into_error_types(self):
        pred_boxes = torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [2.0, 2.0, 3.0, 3.0],
                [4.0, 4.0, 5.0, 5.0],
                [8.0, 8.0, 9.0, 9.0],
            ],
            dtype=torch.float32,
        )
        pred_scores = torch.tensor([0.99, 0.90, 0.80, 0.70, 0.60], dtype=torch.float32)
        pred_labels = torch.tensor([0, 0, 0, 1, 0], dtype=torch.int64)
        gt_boxes = torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0],
                [2.0, 2.0, 4.0, 4.0],
                [4.0, 4.0, 5.0, 5.0],
            ],
            dtype=torch.float32,
        )
        gt_labels = torch.tensor([0, 0, 0], dtype=torch.int64)

        stats = diagnose_image_errors(
            pred_boxes,
            pred_scores,
            pred_labels,
            gt_boxes,
            gt_labels,
            iou_threshold=0.5,
            low_iou=0.1,
            num_classes=2,
            max_dets=100,
        )

        self.assertEqual(stats["matched_gt"], 1)
        self.assertEqual(stats["missed_gt"], 2)
        self.assertEqual(stats["fp"]["duplicate"], 1)
        self.assertEqual(stats["fp"]["localization"], 1)
        self.assertEqual(stats["fp"]["class_confusion"], 1)
        self.assertEqual(stats["fp"]["background"], 1)
        self.assertEqual(stats["per_class"][0]["matched_gt"], 1)
        self.assertEqual(stats["per_class"][0]["missed_gt"], 2)

    def test_iter_image_detections_accepts_deploy_tuple_output(self):
        labels = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
        boxes = torch.tensor(
            [
                [[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]],
                [[4.0, 4.0, 5.0, 5.0], [6.0, 6.0, 7.0, 7.0]],
            ],
            dtype=torch.float32,
        )
        scores = torch.tensor([[0.9, 0.8], [0.7, 0.6]], dtype=torch.float32)

        detections = list(iter_image_detections((labels, boxes, scores)))

        self.assertEqual(len(detections), 2)
        self.assertTrue(torch.equal(detections[0]["labels"], labels[0]))
        self.assertTrue(torch.equal(detections[0]["boxes"], boxes[0]))
        self.assertTrue(torch.equal(detections[0]["scores"], scores[0]))

    def test_iter_image_detections_keeps_dict_output(self):
        expected = [{"labels": torch.tensor([0]), "boxes": torch.zeros(1, 4), "scores": torch.tensor([0.9])}]

        detections = list(iter_image_detections(expected))

        self.assertIs(detections[0], expected[0])


if __name__ == "__main__":
    unittest.main()
