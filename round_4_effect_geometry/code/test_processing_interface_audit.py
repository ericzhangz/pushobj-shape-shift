"""CPU algebra checks before loading the real checkpoint for G2."""

import unittest

import torch

from research.reframe_v3.processing_interface_audit import ObservationProcessor


class ObservationProcessorTests(unittest.TestCase):
    def test_identity_and_explicit_inverse_all_history_slots(self):
        z = {"visual": torch.randn(2, 3, 5, 7, dtype=torch.float64),
             "proprio": torch.randn(2, 3, 4, dtype=torch.float64)}
        identity = ObservationProcessor(0.0)
        shear = ObservationProcessor(0.01)
        for processor in (identity, shear):
            restored = processor.inverse(processor.forward(z))
            for block in z:
                self.assertTrue(torch.allclose(restored[block], z[block], atol=1e-12, rtol=0))
        self.assertGreater((shear.forward(z)["visual"] - z["visual"]).abs().max().item(), 0)


if __name__ == "__main__":
    unittest.main()
