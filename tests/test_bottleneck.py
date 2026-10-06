import unittest

import torch

from concept_retrofit.models import ConceptBottleneck


class BottleneckTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.module = ConceptBottleneck(12, 5, 16, 4, 2, 4)
        self.hidden = torch.randn(2, 3, 12)

    def test_full_residual_is_an_exact_initialization_identity(self):
        reconstructed, _ = self.module(self.hidden, residual_scale=1.0)
        torch.testing.assert_close(reconstructed, self.hidden, atol=1e-6, rtol=1e-6)

    def test_named_scores_receive_language_path_gradients(self):
        reconstructed, _ = self.module(self.hidden, residual_scale=0.5)
        reconstructed.square().mean().backward()
        self.assertGreater(float(self.module.known_encoder.weight.grad.norm()), 0.0)

    def test_intervention_changes_the_reconstruction(self):
        base, _ = self.module(self.hidden, residual_scale=1.0)
        changed, _ = self.module(self.hidden, residual_scale=1.0, interventions={0: 1.0})
        self.assertGreater(float((base - changed).abs().sum()), 0.0)

    def test_signed_controls_have_the_exact_expected_vector_effect(self):
        base, parts = self.module(self.hidden, residual_scale=0.75)
        for value in (2., -1., 4., -3.):
            changed, _ = self.module(self.hidden, residual_scale=0.75, interventions={0: value})
            expected = (value - parts.known_activations[..., 0:1]) * self.module.known_vectors[0]
            torch.testing.assert_close(changed - base, expected, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
