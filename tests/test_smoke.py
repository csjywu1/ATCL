import unittest

import torch

from model.actp import ACTPPredictor


class ACTPSmokeTest(unittest.TestCase):
    def test_forward_shape(self):
        model = ACTPPredictor(hidden=8, bins=5)
        x = torch.zeros(2, 32, 3)
        mask = torch.ones(2, 32, dtype=torch.bool)
        output = model(x, mask)
        self.assertEqual(output["pred"].shape, (2, 5, 2))
        self.assertEqual(len(output["z"]), 3)
        self.assertTrue(torch.isfinite(output["pred"]).all())


if __name__ == "__main__":
    unittest.main()
