"""Bounded Atlas sampling, document-disjoint splits and train-only vocabulary selection."""
import collections
import hashlib
from pathlib import Path
import random

from concept_retrofit.io import digest, rows, save_json

REPOSITORY = 'guidelabs/fineweb-atlas'
REVISION = '4fd517146b77b2431e4170c5e50e064e0a25c5c4'


def partition(document, seed):
    bucket = int(hashlib.sha256(f'{seed}:{document}'.encode()).hexdigest()[:8], 16) % 100
    return 'train' if bucket < 80 else 'validation' if bucket < 90 else 'test'


def bounded_rows(streams, limit):
    """Read shard prefixes fairly and close native scanners even on exceptions."""
    active, scanned = list(streams), 0
    try:
        while active and scanned < limit:
            for stream in list(active):
                try:
                    row = next(stream)
                except StopIteration:
                    active.remove(stream)
                    continue
                scanned += 1
                yield scanned, row
                if scanned >= limit:
                    break
    finally:
        # A suspended parquet reader must be closed before Python shutdown.
        for stream in streams:
            if hasattr(stream, 'close'):
                stream.close()


def prepare(destination, limit=20000, known=64, min_support=10, seed=1729,
            source=None, scan_limit=100000, shards=(0, 17, 35, 53),
            tokenizer=None, max_length=256, training_token_budget=0):
    """Reservoir over a bounded shard prefix. Explicitly NOT full-corpus sampling.

    Local source JSONL uses text, document_id and label_ids. It permits offline
    integration tests and existing Atlas exports without transferring caches.
    """
    import json
    destination = Path(destination)
    if (destination / 'manifest.json').exists():
        raise FileExistsError(f'Dataset already exists: {destination}')
    if limit < 3 or known < 1 or min_support < 1 or scan_limit < limit:
        raise ValueError('Invalid sample, vocabulary, or scan limits')
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {}
    if source:
        streams = [rows(source)]
    else:
        from datasets import load_dataset
        from pyarrow.dataset import ParquetFragmentScanOptions
        metadata = {int(r['concept_id']): dict(r) for r in
                    load_dataset(REPOSITORY, 'concepts', revision=REVISION, split='train')}
        streams = [iter(load_dataset('parquet', data_files=
            f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/fineweb-atlas-annotated/shard_{s:04d}.parquet',
            split='train', streaming=True, batch_size=1024,
            columns=['chunk_text', 'chunk_status', 'doc_int_id', 'content_ids',
                     'tone_ids', 'document_ids', 'entity_ids'],
            fragment_scan_options=ParquetFragmentScanOptions(pre_buffer=False))) for s in shards]
    # Round-robin across bounded shard prefixes; reservoir and IDs are O(limit).
    rng, sample, scanned, accepted = random.Random(seed), [], 0, 0
    from contextlib import closing
    with closing(bounded_rows(streams, scan_limit)) as records:
        for scanned, raw in records:
            if scanned % max(1, scan_limit // 100) == 0:
                print(f'PROGRESS prepare scan={scanned}/{scan_limit} sampled={len(sample)}', flush=True)
            text = raw.get('text', raw.get('chunk_text', '')).strip()
            if not text or raw.get('chunk_status', 'ok') != 'ok':
                continue
            token_count = len(tokenizer.encode(text, add_special_tokens=False)) if tokenizer else None
            if token_count is not None and not 2 <= token_count <= max_length:
                continue
            labels = raw.get('label_ids', raw.get('atlas_ids'))
            if labels is None:
                labels = [c for field in ('content_ids', 'tone_ids', 'document_ids', 'entity_ids')
                          for c in (raw.get(field) or [])]
            document = raw.get('document_id', raw.get('doc_int_id'))
            if document is None:
                raise ValueError('Every input requires its parent document_id for safe splitting')
            record = {'text': text, 'document_id': str(document),
                      'label_ids': sorted(set(map(int, labels))), 'token_count': token_count,
                      'id': hashlib.sha256(text.encode()).hexdigest()}
            accepted += 1
            if len(sample) < limit:
                sample.append(record)
            else:
                index = rng.randrange(accepted)
                if index < limit:
                    sample[index] = record
    streams.clear()
    import gc
    gc.collect()
    # Identical text is assigned once; sibling chunks use parent document splits.
    sample = list({r['id']: r for r in sample}.values())
    frequency = collections.Counter(c for r in sample if partition(r['document_id'], seed) == 'train'
                                    for c in r['label_ids'])
    groups = collections.defaultdict(list)
    for c, n in frequency.items():
        if n >= min_support:
            m = metadata.get(c, {})
            groups[(str(m.get('concept_type', 'local')), str(m.get('taxonomy_lcc_path_primary') or ''))].append(c)
    for group in groups.values():
        group.sort(key=lambda c: (-frequency[c], c))
    selected = []
    while len(selected) < known and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(selected) < known:
                selected.append(groups[key].pop(0))
    if len(selected) != known:
        raise ValueError(f'Only {len(selected)} concepts have {min_support}+ train positives; '
                         'increase scan/sample limits or lower --known')
    selected_set = set(selected)
    unique_train_tokens = sum((r['token_count'] or 0) - 1 for r in sample
                             if partition(r['document_id'], seed) == 'train' and r['token_count'])
    if training_token_budget and unique_train_tokens < training_token_budget:
        raise ValueError(f'Sample has {unique_train_tokens:,} unique training target tokens; '
                         f'{training_token_budget:,} required. Increase sample/scan limits.')
    counts, positives = {}, {}
    for split in ('train', 'validation', 'test'):
        split_rows = [r for r in sample if partition(r['document_id'], seed) == split]
        if not split_rows:
            raise ValueError(f'Empty {split} split: increase sample size')
        # Keep unlabeled rows for language modeling; never filter only test positives.
        with (destination / f'{split}.jsonl').open('w') as stream:
            for r in split_rows:
                r['label_ids'] = [c for c in r['label_ids'] if c in selected_set]
                stream.write(json.dumps(r) + '\n')
        counts[split] = len(split_rows)
        support = collections.Counter(c for r in split_rows for c in r['label_ids'])
        positives[split] = {str(c): support[c] for c in selected}
    save_json(destination / 'manifest.json', {
        'schema_version': 1, 'dataset': str(source) if source else REPOSITORY,
        'revision': None if source else REVISION, 'seed': seed, 'concept_ids': selected,
        'concepts': [metadata.get(c, {'concept_id': c, 'name': f'concept_{c}'}) for c in selected],
        'counts': counts, 'positive_counts': positives,
        'unique_training_target_tokens': unique_train_tokens,
        'tokenizer': getattr(tokenizer, 'name_or_path', None), 'max_length': max_length,
        'sampling': 'Reservoir over bounded round-robin shard prefixes; not uniform full-corpus sampling',
        'scanned': scanned, 'accepted': accepted, 'shards': list(shards) if not source else [],
        'label_policy': 'Machine annotation membership; omitted labels are unassigned, not verified semantic negatives',
        'sha256': {s: digest(destination / f'{s}.jsonl') for s in counts}})


class Corpus:
    """Validate manifests and load bounded JSONL splits, without activation caches."""
    def __init__(self, directory):
        import json
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / 'manifest.json').read_text())
        self.ids = self.manifest['concept_ids']
        self.index = {c: i for i, c in enumerate(self.ids)}
        if len(self.index) != len(self.ids):
            raise ValueError('Duplicate concept IDs')
        self.splits = {}
        documents = set()
        texts = set()
        for split in ('train', 'validation', 'test'):
            path = self.directory / f'{split}.jsonl'
            if digest(path) != self.manifest['sha256'][split]:
                raise ValueError(f'Corrupted split {split}')
            values = list(rows(path))
            if not values:
                raise ValueError(f'Empty split {split}')
            current = {r['document_id'] for r in values}
            current_texts = {r['id'] for r in values}
            if current & documents or current_texts & texts:
                raise ValueError('Document/text leakage between splits')
            documents.update(current)
            texts.update(current_texts)
            self.splits[split] = values

    def batch(self, tokenizer, values, max_length, device):
        import torch
        # Never assign whole-chunk labels to a truncated prefix.
        texts = [r['text'] for r in values]
        full = tokenizer(texts, add_special_tokens=False)['input_ids']
        inputs = tokenizer(texts, add_special_tokens=False, padding=True, truncation=True,
                           max_length=max_length, return_tensors='pt').to(device)
        labels = torch.zeros(len(values), len(self.ids), device=device)
        label_valid = torch.tensor([len(x) <= max_length for x in full], device=device)
        for i, row in enumerate(values):
            for c in row['label_ids']:
                if c in self.index:
                    labels[i, self.index[c]] = 1.
        return inputs, labels, label_valid
