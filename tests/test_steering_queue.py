"""Queue must not run until the main stages finish and the trainer exits."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('steering_after_main', Path(__file__).resolve().parents[1]/'scripts/steering_after_main.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class QueueTests(unittest.TestCase):
    def test_waits_for_complete_lora_and_parent_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for stage in ('frozen','lora'):
                (root/stage).mkdir()
                (root/stage/'status.json').write_text(json.dumps({'state':'complete'}))
            (root/'lora/best.pt').write_bytes(b'checkpoint')
            with patch.object(queue, 'process_alive', return_value=True):
                self.assertEqual(queue.parent_state(root,123)[0], 'waiting')
            with patch.object(queue, 'process_alive', return_value=False):
                self.assertEqual(queue.parent_state(root,123)[0], 'ready')
                (root/'lora/status.json').write_text(json.dumps({'state':'running'}))
                self.assertEqual(queue.parent_state(root,123)[0], 'parent_stopped')


class MergedScheduleTests(unittest.TestCase):
    def test_evaluation_spans_32_supported_content_controls(self):
        spec = importlib.util.spec_from_file_location('merged_retrofit', Path(__file__).resolve().parents[1]/'scripts/merged_retrofit.py')
        merged = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(merged)
        manifest = {'concepts': [
            {'name':f'Topic {i}', 'concept_type':'content',
             'taxonomy_lcc_primary':f'Domain {i % 8}'} for i in range(40)]}
        resource = {'concepts':{str(i):{} for i in range(40)}}
        names = merged.evaluation_concepts(manifest, resource)
        self.assertEqual(len(names),32)
        self.assertEqual(len(set(names)),32)
        self.assertEqual(len({int(name.split()[-1]) % 8 for name in names}),8)
        del resource['concepts']['0']
        self.assertNotIn('Topic 0',merged.evaluation_concepts(manifest,resource))

    def test_four_steering_phases_keep_capability_training_after_them(self):
        spec = importlib.util.spec_from_file_location('merged_retrofit', Path(__file__).resolve().parents[1]/'scripts/merged_retrofit.py')
        merged = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(merged)
        phases = merged.phase_plan(5,2,7)
        self.assertEqual([stage for _,stage,_ in phases], ['steering','lora']*4)
        self.assertEqual(sum(n for _,stage,n in phases if stage=='steering'),20)
        self.assertEqual(phases[-1],('capability-04','lora',7))
