"""Selected-face inference contract, without training or loading real weights."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch

from emotion_inference import DanEmotionPredictor


def attention(*, state="locked", observed=True, face=True, target_id=7):
    mesh = NS(landmark=[NS(x=float(x), y=float(y))
                       for x, y in zip(np.linspace(0.25, 0.75, 468),
                                       np.linspace(0.1, 0.4, 468))])
    target = NS(track_id=7, observed=observed, bbox=np.array([40, 20, 240, 220]))
    other = NS(track_id=99, observed=True, bbox=np.array([260, 20, 390, 220]))
    tracking = NS(state=state, target_id=target_id, pose=object(), tracks=[other, target])
    return NS(tracking=tracking, crop_rect=(40, 20, 200, 200),
              holistic=NS(face_landmarks=mesh if face else None),
              hand_boxes=[], playing=None)


class FakeModel(torch.nn.Module):
    def __init__(self, values=(0.0, 1.0)):
        super().__init__()
        self.values = values
        self.inputs = []
        self.load_calls = []

    def load_state_dict(self, state, strict=True):
        self.load_calls.append((state, strict))

    def forward(self, image):
        self.inputs.append(image.detach().cpu().clone())
        return torch.tensor([self.values], dtype=torch.float32, device=image.device), None, None


class EmotionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.weights = Path(self.temp.name) / "weights.pth"
        self.weights.write_bytes(b"mock checkpoint")
        self.frame = np.zeros((260, 400, 3), dtype=np.uint8)
        self.frame[:] = (10, 80, 240)

    def predictor(self, values=(0.0, 1.0), **options):
        model = FakeModel(values)
        with patch("emotion_inference.DAN", return_value=model) as constructor, \
                patch("emotion_inference.torch.load", return_value={"mock": 1}) as load:
            predictor = DanEmotionPredictor(self.weights, device="cpu", **options)
        constructor.assert_called_once_with(num_class=2, num_head=4, pretrained=False)
        load.assert_called_once_with(self.weights.resolve(), map_location="cpu", weights_only=True)
        self.assertEqual(model.load_calls, [({"mock": 1}, True)])
        self.assertFalse(model.training)
        return predictor, model

    def test_positive_and_bgr_to_rgb_preprocessing_without_hands(self):
        predictor, model = self.predictor()
        result = predictor.process(self.frame, attention())
        self.assertEqual(result.label, 1)
        self.assertAlmostEqual(result.probability, 0.7310586, places=6)
        image = model.inputs[0]
        self.assertEqual(tuple(image.shape), (1, 3, 224, 224))
        self.assertEqual(image.dtype, torch.float32)
        expected = (np.array([240, 80, 10]) / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        np.testing.assert_allclose(image[0, :, 100, 100].numpy(), expected, atol=1e-6)
        x1, y1, x2, y2 = result.face_bbox
        self.assertGreaterEqual(x1, 40)
        self.assertGreaterEqual(y1, 20)
        self.assertLessEqual(x2, 240)
        self.assertLessEqual(y2, 220)
        self.assertLess((x2 - x1) * (y2 - y1), 200 * 200)

    def test_equal_probability_and_lower_probability_are_neutral(self):
        for logits, probability in [((1, 1), 0.5), ((1, 0), 0.2689414)]:
            with self.subTest(logits=logits):
                predictor, _ = self.predictor(logits)
                result = predictor.process(self.frame, attention())
                self.assertEqual(result.label, 0)
                self.assertAlmostEqual(result.probability, probability, places=6)

    def test_missing_and_untrusted_targets_do_not_run_model(self):
        cases = [attention(face=False), attention(state="recovering"),
                 attention(observed=False), attention(target_id=100),
                 attention(target_id=None)]
        for item in cases:
            with self.subTest(item=item):
                predictor, model = self.predictor()
                result = predictor.process(self.frame, item)
                self.assertEqual(result.label, 2)
                self.assertIsNone(result.probability)
                self.assertIsNone(result.face_bbox)
                self.assertEqual(model.inputs, [])

    def test_face_in_crop_for_another_track_is_rejected(self):
        predictor, model = self.predictor()
        item = attention()
        item.crop_rect = (260, 20, 130, 200)
        result = predictor.process(self.frame, item)
        self.assertEqual(result.label, 2)
        self.assertEqual(result.detail, "face_outside_selected_target")
        self.assertFalse(model.inputs)

    def test_bad_mesh_and_tiny_face_are_missing(self):
        for coordinates in [[(float("nan"), 0.2)] * 468,
                            [(1.3, 0.2)] * 468,
                            [(0.3, 0.3)] * 468,
                            [(0.2, 0.2)] * 10]:
            with self.subTest(coordinates=coordinates[:1]):
                predictor, model = self.predictor()
                item = attention()
                item.holistic.face_landmarks.landmark = [NS(x=x, y=y) for x, y in coordinates]
                self.assertEqual(predictor.process(self.frame, item).label, 2)
                self.assertFalse(model.inputs)

    def test_invalid_crop_is_missing(self):
        for rectangle in [None, (-1, 20, 200, 200), (40, 20, 400, 200),
                          (40, 20, 0, 200), (40, 20, float("inf"), 200)]:
            predictor, model = self.predictor()
            item = attention()
            item.crop_rect = rectangle
            self.assertEqual(predictor.process(self.frame, item).label, 2)
            self.assertFalse(model.inputs)

    def test_mismatched_visible_holistic_nose_is_rejected(self):
        predictor, model = self.predictor()
        item = attention()
        item.holistic.pose_landmarks = NS(landmark=[NS(x=0.9, y=0.9, visibility=1.0)])
        result = predictor.process(self.frame, item)
        self.assertEqual(result.label, 2)
        self.assertEqual(result.detail, "face_pose_mismatch")
        self.assertFalse(model.inputs)

    def test_model_failures_are_not_changed_into_missing(self):
        predictor, _ = self.predictor((float("nan"), 0.0))
        with self.assertRaisesRegex(RuntimeError, "invalid binary logits"):
            predictor.process(self.frame, attention())
        predictor, model = self.predictor()
        with patch.object(model, "forward", side_effect=RuntimeError("material model failure")):
            with self.assertRaisesRegex(RuntimeError, "material model failure"):
                predictor.process(self.frame, attention())

    def test_missing_weights_fail_before_model_construction(self):
        with patch("emotion_inference.DAN") as constructor:
            with self.assertRaises(FileNotFoundError):
                DanEmotionPredictor(Path(self.temp.name) / "absent.pth")
            constructor.assert_not_called()

    def test_explicit_unavailable_cuda_raises(self):
        with patch("emotion_inference.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
                DanEmotionPredictor(self.weights, device="cuda")

    def test_auto_uses_cpu_when_cuda_unavailable(self):
        with patch("emotion_inference.DAN", return_value=FakeModel()), \
                patch("emotion_inference.torch.load", return_value={}), \
                patch("emotion_inference.torch.cuda.is_available", return_value=False):
            predictor = DanEmotionPredictor(self.weights)
        self.assertEqual(predictor.device.type, "cpu")
        self.assertEqual(predictor.process(self.frame, attention()).label, 1)


if __name__ == "__main__":
    unittest.main()
