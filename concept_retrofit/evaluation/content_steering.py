"""Standalone matched steering audit without stopping an ongoing trainer."""
import json
from pathlib import Path
import shutil
import torch

from concept_retrofit.io import save_json
from concept_retrofit.pipeline import load_trained
from .steering import generate_with_model
from .judge import judge_folder


def run(data, checkpoint, output, names, max_new_tokens=96):
    folder = Path(output)
    if folder.exists():
        raise FileExistsError('Use a fresh audit folder; saved examples must not be silently overwritten')
    folder.mkdir(parents=True)
    # Atomic checkpoint replacement leaves an opened source inode intact.
    snapshot = folder / 'checkpoint.pt'
    shutil.copyfile(checkpoint, snapshot)
    cfg, corpus, model, tokenizer, module = load_trained(data, snapshot)
    concepts = {c['name'].casefold(): (i, c) for i, c in enumerate(corpus.manifest['concepts'])}
    selected = []
    for name in names:
        index, metadata = concepts[name.casefold()]
        if metadata.get('concept_type') != 'content':
            raise ValueError(f'{name} is not a content concept')
        selected.append({'concept_index': index, 'concept_id': corpus.ids[index], 'metadata': metadata})
    state = torch.load(snapshot, map_location='cpu', weights_only=True)
    save_json(folder / 'audit.json', {'source_checkpoint': str(checkpoint),
        'step': state['step'], 'training_tokens': state['tokens_seen'], 'stage': state['stage'],
        'residual_scale': cfg.residual_scale, 'max_new_tokens': max_new_tokens,
        'concepts': selected, 'state': 'running'})
    del state
    save_json(folder / 'interventions.json', {'concepts': selected})
    for concept in selected:
        metadata = concept['metadata']
        print(f'CONTENT_STEERING concept={metadata["name"]} index={concept["concept_index"]}', flush=True)
        prompts = ['Write a short story about someone spending an evening at home. Keep it under 70 words.',
                   f'Write a short scene involving {metadata["name"]}. '
                   f'The topic means: {metadata["description"]} Keep it under 70 words.']
        generate_with_model(cfg, corpus, model, tokenizer, module,
                            folder / f'steering-concept-{concept["concept_index"]}.json',
                            max_new_tokens, concept['concept_index'], prompts)
    # Free this audit's GPU resources before the paced remote judge calls.
    del model, module
    import gc
    gc.collect()
    if cfg.device.startswith('cuda'):
        torch.cuda.empty_cache()
    summary = judge_folder(folder)
    report = json.loads((folder / 'audit.json').read_text())
    report['state'] = 'complete'
    save_json(folder / 'audit.json', report)
    print(f'CONTENT_STEERING_COMPLETE out={folder}', flush=True)
    return summary
