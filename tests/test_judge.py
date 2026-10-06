import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from concept_retrofit.evaluation.judge import build_request, validate_response, judge_folder
from concept_retrofit.io import save_json


def response(mapping):
    presence = {'base': 1, 'retrofit': 2, 'amplify': 4, 'suppress': 0}
    scores = [{'candidate_id': name, 'target_presence': presence[condition],
               'fluency': 3, 'prompt_following': 4, 'unrelated_meaning_preservation': 3,
               'rationale': 'Grounded in the supplied text.'}
              for name, condition in mapping.items()]
    return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'scores': scores})}}]}


def fixture(folder):
    metadata = {'name': 'Cooking', 'description': 'Preparing food.'}
    save_json(folder / 'interventions.json', {'concepts': [
        {'concept_index': 0, 'concept_id': 17, 'metadata': metadata}]})
    examples = {name: text for name, text in zip(('base', 'retrofit', 'amplify', 'suppress'),
        ('A quiet afternoon.', 'She prepared soup.', 'She cooked soup and baked bread.', 'She read a book.'))}
    save_json(folder / 'steering-concept-0.json', {'examples': [
        {'prompt': 'Write a story.', 'condition': name, 'response': text}
        for name, text in examples.items()]})
    return metadata, examples


class JudgeTests(unittest.TestCase):
    def test_blinding_and_strict_schema(self):
        examples = {name: name + ' text' for name in ('base', 'retrofit', 'amplify', 'suppress')}
        payload, mapping = build_request('A story.', examples, {'name': 'Cooking'})
        candidates = json.loads(payload['messages'][1]['content'])['candidates']
        self.assertTrue(all(set(c) == {'candidate_id', 'text'} for c in candidates))
        scores = validate_response(response(mapping), mapping)
        self.assertEqual(scores['amplify']['target_presence'], 4)
        bad = response(mapping)
        bad['choices'][0]['finish_reason'] = 'length'
        with self.assertRaises(ValueError):
            validate_response(bad, mapping)
        bad = response(mapping)
        parsed = json.loads(bad['choices'][0]['message']['content'])
        parsed['scores'][0]['fluency'] = True
        bad['choices'][0]['message']['content'] = json.dumps(parsed)
        with self.assertRaises(ValueError):
            validate_response(bad, mapping)

    def test_scores_cache_and_missing_key(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            metadata, examples = fixture(folder)
            _, mapping = build_request('Write a story.', examples, metadata)
            calls = []
            def fake_request(payload, key, interval):
                calls.append(interval)
                self.assertEqual(key, 'test-secret')
                return response(mapping)
            with patch('concept_retrofit.evaluation.judge.resolve_key', return_value='test-secret'):
                summary = judge_folder(folder, request_fn=fake_request)
                self.assertEqual(summary['amplify']['direction_success_rate'], 1.)
                self.assertEqual(summary['suppress']['direction_success_rate'], 1.)
                judge_folder(folder, request_fn=fake_request)
            self.assertEqual(calls, [10.])
            self.assertNotIn('test-secret', ''.join(p.read_text() for p in folder.rglob('*.json')))
            with tempfile.TemporaryDirectory() as empty:
                fixture(Path(empty))
                with patch('concept_retrofit.evaluation.judge.resolve_key', return_value=''):
                    skipped = judge_folder(empty, request_fn=fake_request)
                self.assertEqual(skipped['skipped_groups'], 1)
                self.assertIsNone(skipped['amplify']['direction_success_rate'])

    def test_invalid_responses_are_errors_not_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            fixture(folder)
            with patch('concept_retrofit.evaluation.judge.resolve_key', return_value='test-secret'):
                summary = judge_folder(folder, request_fn=lambda *args: {'error': 'bad response'})
            self.assertEqual(summary['failed_groups'], 1)
            self.assertEqual(summary['valid_groups'], 0)
            self.assertIsNone(summary['suppress']['direction_success_rate'])


if __name__ == '__main__':
    unittest.main()
