"""Causal alignment, signed suppression, hook cleanup and real Qwen training."""
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from concept_retrofit.pipeline import RunConfig, train_stage, load_trained
from concept_retrofit.training.steering import (calibrated_direction, suppress_logits,
    injection_positions, intervention_losses, build_lexicon)
from concept_retrofit.models.bottleneck import ConceptBottleneck
from concept_retrofit.models.qwen import load_text
from concept_retrofit.evaluation.steering import generate_examples
from test_pipeline import fixture


class SteeringTrainingTests(unittest.TestCase):
    def test_calibration_and_suppression_do_not_promote_anti_aligned_tokens(self):
        class Model:
            head = nn.Linear(2, 3, bias=False)
        model = Model()
        with torch.no_grad():
            model.head.weight.copy_(torch.tensor([[2., 0.], [-3., 0.], [0., 1.]]))
        module = ConceptBottleneck(2, 1, 2, 1, 1, 1)
        with torch.no_grad():
            module.known_vectors[0].copy_(torch.tensor([4., 0.]))
        injected, unit, peak = calibrated_direction(model, module, 0, .1)
        self.assertEqual(peak, 2.)
        torch.testing.assert_close((model.head(injected)).max(), torch.tensor(.1))
        logits = torch.tensor([[5., 5., 5.]])
        suppressed = suppress_logits(logits, model.head, unit, .5)
        torch.testing.assert_close(suppressed, torch.tensor([[4., 5., 5.]]))

    def test_causal_mask_excludes_padding_and_unassigned_chunks(self):
        inputs = {'input_ids': torch.tensor([[4, 3, 4, 0], [4, 3, 4, 0]]),
                  'attention_mask': torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]])}
        positions = injection_positions(inputs, torch.tensor([[1.], [0.]]),
                                        torch.tensor([True, True]), 0, [3, 0])
        self.assertEqual(positions.tolist(), [[True, False, False, False], [False]*4])

    def test_hook_cleanup_teacher_invariance_and_train_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            _, model_path = fixture(directory)
            model, tokenizer = load_text(str(model_path), device='cpu')
            inputs = tokenizer('cook a story', return_tensors='pt')
            original = model.hidden(inputs).detach()
            positions = torch.tensor([[False, False, True]])
            with model.inject(torch.ones(16)*.1, positions, 1):
                modified = model.hidden(inputs)
            self.assertGreater(float((modified-original).abs().sum()), 0)
            torch.testing.assert_close(model.hidden(inputs), original, atol=0, rtol=0)
            with self.assertRaisesRegex(RuntimeError, 'test interruption'):
                with model.inject(torch.ones(16)*.1, positions, 1):
                    raise RuntimeError('test interruption')
            self.assertFalse(model.backbone.layers[1]._forward_pre_hooks)
            model.add_lora(1, 2)
            module = ConceptBottleneck(16, 2, 8, 4, 1, 2)
            direction, _, _ = calibrated_direction(model, module, 0, .1)
            with model.inject(direction, positions, 1):
                hidden = model.hidden(inputs)
            reconstructed, parts = module(hidden, .75)
            respond, express = intervention_losses(model, parts, reconstructed, positions, 0, [3])
            (respond+express).backward()
            self.assertGreater(float(module.known_encoder.weight.grad.abs().sum()), 0)
            self.assertGreater(sum(float(p.grad.abs().sum()) for n,p in model.named_parameters()
                                   if n.endswith('.b') and p.grad is not None), 0)

    def test_real_qwen_steering_stage_saves_and_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, model_path = fixture(root)
            from concept_retrofit.io import digest
            # Place the concept token after a causal context token; a concept
            # at position zero has no preceding prediction state to supervise.
            for split in ('train', 'validation', 'test'):
                path = data/f'{split}.jsonl'
                values = [json.loads(line) for line in path.read_text().splitlines()]
                for row in values:
                    row['text'] = 'a ' + row['text']
                path.write_text(''.join(json.dumps(row)+'\n' for row in values))
            manifest = json.loads((data/'manifest.json').read_text())
            manifest['sha256'] = {split: digest(data/f'{split}.jsonl') for split in ('train', 'validation', 'test')}
            (data/'manifest.json').write_text(json.dumps(manifest))
            cfg = RunConfig(model=str(model_path), revision=None, device='cpu', steps=2,
                evaluate_every=1, evaluation_documents=4, batch_size=4,
                unknown_features=8, unknown_rank=4, known_topk=1, unknown_topk=2,
                max_length=16, top_layers=1, lora_rank=2,
                maximum_nll_increase=None, maximum_validation_kl=None)
            frozen = train_stage(data, root/'frozen', cfg)
            model, tokenizer = load_text(str(model_path), device='cpu')
            lexicon = root/'lexicon.json'
            entries = build_lexicon(data, tokenizer, lexicon, [0,1],
                                    max_documents=100, min_count=2, minimum_lift=1.1)
            self.assertEqual(len(entries), 2)
            steering_cfg = replace(cfg, steering_mode='layer', steering_lexicon=str(lexicon), steering_tau=.1)
            checkpoint = train_stage(data, root/'steering', steering_cfg, 'steering', frozen)
            _, _, trained, _, module = load_trained(data, checkpoint)
            state = torch.load(checkpoint, weights_only=True)
            self.assertTrue(state['steering_lexicon_sha256'])
            self.assertTrue(state['adapters'])
            self.assertGreater(state['steering_counts']['positions'], 0)
            generate_examples(data, checkpoint, root/'examples.json', max_new_tokens=2)
            examples = json.loads((root/'examples.json').read_text())
            self.assertEqual(examples['intervention_method'], 'layer')
            self.assertEqual(examples['examples'][2]['injection_tau'], .1)
            self.assertEqual(examples['examples'][3]['suppression_logit_strength'], 1.)
            self.assertFalse(any(b._forward_pre_hooks for b in trained.backbone.layers))


if __name__ == '__main__':
    unittest.main()
