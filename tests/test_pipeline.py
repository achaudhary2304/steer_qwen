"""Offline integration tests using an actual tiny Qwen3.5 text model."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from concept_retrofit.data.prepare import Corpus, prepare
from concept_retrofit.pipeline import RunConfig, train_stage, evaluate_checkpoint, probe_layers
from concept_retrofit.evaluation.steering import generate_examples
from concept_retrofit.models.qwen import load_text


def fixture(directory):
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    directory = Path(directory)
    model_path = directory / 'model'
    torch.manual_seed(17)
    config = Qwen3_5TextConfig(vocab_size=24, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8,
        linear_value_head_dim=8, layer_types=['linear_attention', 'full_attention'],
        pad_token_id=0, eos_token_id=1, max_position_embeddings=128)
    Qwen3_5ForCausalLM(config).save_pretrained(model_path)
    vocab = {'[PAD]': 0, '[EOS]': 1, '[UNK]': 2, 'cook': 3, 'music': 4,
             'story': 5, 'a': 6, 'quiet': 7, 'afternoon': 8}
    backend = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]',
                                       eos_token='[EOS]', unk_token='[UNK]')
    tokenizer.save_pretrained(model_path)
    source = directory / 'source.jsonl'
    with source.open('w') as stream:
        for i in range(100):
            stream.write(json.dumps({'text': f'{"cook" if i % 2 else "music"} a story {i}',
                'document_id': i, 'label_ids': [i % 2]}) + '\n')
    data = directory / 'data'
    prepare(data, limit=100, known=2, min_support=5, source=source, scan_limit=100, tokenizer=tokenizer)
    return data, model_path


class PipelineTests(unittest.TestCase):
    def test_real_qwen_end_to_end_and_teacher_invariance(self):
        with tempfile.TemporaryDirectory() as directory:
            data, model_path = fixture(directory)
            cfg = RunConfig(model=str(model_path), revision=None, device='cpu',
                steps=2, evaluate_every=1, batch_size=2, evaluation_documents=12,
                unknown_features=8, unknown_rank=4, known_topk=1, unknown_topk=2,
                max_length=16, top_layers=2, lora_rank=2,
                maximum_nll_increase=100, maximum_validation_kl=100)
            out = Path(directory) / 'runs'
            probe_layers(data, cfg, out / 'probes.json')
            frozen = train_stage(data, out / 'frozen', cfg)
            lora = train_stage(data, out / 'lora', cfg, 'lora', frozen)
            report = evaluate_checkpoint(data, lora, out / 'test.json')
            self.assertGreater(report['tokens'], 0)
            self.assertIn('residual', report['leakage_probes'])
            generate_examples(data, lora, out / 'examples.json', max_new_tokens=2)
            self.assertEqual(len(json.loads((out / 'examples.json').read_text())['examples']), 8)
            saved = torch.load(lora, weights_only=True)
            self.assertTrue(saved['adapters'])
            self.assertTrue(any('linear_attn' in name for name in saved['adapters']))
            self.assertGreater(sum(float(v.abs().sum()) for n, v in saved['adapters'].items() if n.endswith('.b')), 0)
            # Frozen base logits must survive arbitrary changes to LoRA weights.
            model, tokenizer = load_text(str(model_path), device='cpu')
            inputs = tokenizer('cook a story', return_tensors='pt')
            with torch.no_grad():
                original = model.hidden(inputs).clone()
            model.add_lora(2, 2)
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name.endswith('.b'):
                        parameter.fill_(0.1)
                with model.teacher():
                    teacher = model.hidden(inputs)
                torch.testing.assert_close(teacher, original, atol=0, rtol=0)

    def test_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as directory:
            data, model_path = fixture(directory)
            cfg = RunConfig(model=str(model_path), revision=None, device='cpu', steps=4,
                evaluate_every=2, evaluation_documents=4, unknown_features=8,
                unknown_rank=4, known_topk=1, unknown_topk=2, max_length=16,
                maximum_nll_increase=100, maximum_validation_kl=100)
            root = Path(directory)
            train_stage(data, root / 'full', cfg)
            from concept_retrofit import pipeline
            original_save = pipeline.save_json
            def interrupt(path, value):
                original_save(path, value)
                if str(path).endswith('status.json') and value['step'] == 2:
                    raise RuntimeError('simulated interruption')
            with patch.object(pipeline, 'save_json', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    train_stage(data, root / 'resumed', cfg)
            train_stage(data, root / 'resumed', cfg, resume=True)
            full = torch.load(root / 'full/last.pt', weights_only=True)
            resumed = torch.load(root / 'resumed/last.pt', weights_only=True)
            for name in full['bottleneck']:
                torch.testing.assert_close(full['bottleneck'][name], resumed['bottleneck'][name], atol=0, rtol=0)

    def test_split_integrity_and_token_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            data, _ = fixture(directory)
            Corpus(data)
            with (data / 'test.jsonl').open('a') as stream:
                stream.write('{}\n')
            with self.assertRaisesRegex(ValueError, 'Corrupted split'):
                Corpus(data)

    def test_failed_gate_preserves_checkpoint_and_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            data, model_path = fixture(directory)
            cfg = RunConfig(model=str(model_path), revision=None, device='cpu', steps=2,
                evaluate_every=1, evaluation_documents=4, unknown_features=8,
                unknown_rank=4, known_topk=1, unknown_topk=2, max_length=16,
                maximum_validation_kl=0.0)
            output = Path(directory) / 'failed'
            with self.assertRaisesRegex(RuntimeError, 'Validation gate failed'):
                train_stage(data, output, cfg)
            self.assertTrue((output / 'last.pt').exists())
            self.assertEqual(json.loads((output / 'status.json').read_text())['state'], 'gate_failed')
            cfg.training_tokens = 100000000
            with self.assertRaisesRegex(ValueError, 'unique Atlas-token budget'):
                train_stage(data, Path(directory) / 'insufficient-data', cfg)


if __name__ == '__main__':
    unittest.main()
