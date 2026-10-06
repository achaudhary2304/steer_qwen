"""Steerling-inspired causal interventions and train-only weak token labels.

Atlas supplies chunk labels. Lift-based token locations are a documented
approximation, NOT the authors' token-level annotated dataset.
"""
import collections
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from concept_retrofit.io import digest, save_json


def calibrated_direction(model, module, concept, tau):
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError('tau must be finite and positive')
    direction = F.normalize(module.known_vectors[concept].float(), dim=0)
    # Freeze the direction during backbone intervention training: the concept
    # vector still learns through the ordinary bottleneck prediction pathway.
    direction = direction.detach()
    peak = (model.head.weight @ direction.to(model.head.weight.dtype)).float().max()
    if not torch.isfinite(peak) or peak <= 1e-8:
        raise ValueError('Concept direction has no positive vocabulary alignment')
    return direction * (tau / peak), direction, float(peak)


def suppress_logits(logits, head, direction, strength, positive_alignment=None):
    if not math.isfinite(strength) or strength < 0:
        raise ValueError('Suppression strength must be finite and nonnegative')
    if positive_alignment is None:
        positive_alignment = (head.weight @ direction.to(head.weight.dtype)).float().relu()
    return logits - strength * positive_alignment


def build_lexicon(data, tokenizer, output, concept_indices, max_documents=20000,
                  top_tokens=64, min_count=5, minimum_lift=2.):
    from concept_retrofit.data.prepare import Corpus
    corpus = Corpus(data)
    selected = set(concept_indices)
    if not selected or not selected <= set(range(len(corpus.ids))):
        raise ValueError('Concept indices must be nonempty and inside the vocabulary')
    if max_documents < 1 or top_tokens < 1 or min_count < 1 or minimum_lift <= 0:
        raise ValueError('Invalid lexical resource limits')
    global_counts, counts = collections.Counter(), {c: collections.Counter() for c in selected}
    totals = collections.Counter()
    records = corpus.splits['train'][:max_documents]
    for offset in range(0, len(records), 256):
        batch = records[offset:offset+256]
        encoded = tokenizer([r['text'] for r in batch], add_special_tokens=False)['input_ids']
        for row, ids in zip(batch, encoded):
            global_counts.update(ids)
            for atlas_id in row['label_ids']:
                c = corpus.index.get(atlas_id)
                if c in selected:
                    counts[c].update(ids); totals[c] += len(ids)
        if offset % 2048 == 0:
            print(f'STEERING_LEXICON documents={min(offset+256,len(records))}/{len(records)}', flush=True)
    total = sum(global_counts.values())
    special = set(tokenizer.all_special_ids)
    entries = {}
    for c in sorted(selected):
        ranked = []
        for token, n in counts[c].items():
            text = tokenizer.decode([token]).strip()
            lift = (n / max(1, totals[c])) / (global_counts[token] / max(1, total))
            # Avoid punctuation and single-character fragments as proxy labels.
            if token not in special and n >= min_count and lift >= minimum_lift and sum(ch.isalpha() for ch in text) >= 3:
                ranked.append((lift, n, token))
        ranked.sort(reverse=True)
        chosen = ranked[:top_tokens]
        if chosen:
            entries[str(c)] = {'atlas_id': corpus.ids[c], 'name': corpus.manifest['concepts'][c]['name'],
                'tokens': [t for _, _, t in chosen],
                'token_text': [tokenizer.decode([t]) for _, _, t in chosen],
                'lift': [lift for lift, _, _ in chosen]}
    if not entries:
        raise ValueError('No eligible concept tokens; increase training sample or lower support')
    save_json(output, {'manifest_sha256': digest(Path(data)/'manifest.json'),
        'model': tokenizer.name_or_path, 'documents': len(records), 'split': 'train',
        'annotation_policy': 'Weak token positions: chunk membership AND train-only high-lift target token; not human token annotations',
        'top_tokens': top_tokens, 'min_count': min_count, 'minimum_lift': minimum_lift,
        'concepts': entries})
    return entries


def injection_positions(inputs, labels, label_valid, concept, tokens):
    ids, mask = inputs['input_ids'], inputs['attention_mask'].bool()
    positions = torch.zeros_like(mask)
    token_ids = torch.tensor(tokens, device=ids.device)
    # Shift annotations: the state at t predicts the annotated target at t+1.
    positions[:, :-1] = (torch.isin(ids[:, 1:], token_ids) & mask[:, :-1] & mask[:, 1:]
                         & labels[:, concept:concept+1].bool() & label_valid[:, None])
    return positions


def intervention_losses(model, parts, reconstructed, positions, concept, tokens, block=16):
    if not positions.any():
        zero = reconstructed.sum() * 0
        return zero, zero
    respond = F.softplus(-parts.known_logits[..., concept][positions]).mean()
    states = reconstructed[positions]
    selected = torch.tensor(tokens, device=states.device)
    express = states.sum() * 0
    from torch.utils.checkpoint import checkpoint
    def token_loss(h):
        logits = model.logits(h)
        return (torch.logsumexp(logits, -1) - torch.logsumexp(logits[:, selected], -1)).sum()
    for offset in range(0, len(states), block):
        express = express + checkpoint(token_loss, states[offset:offset+block], use_reentrant=False)
    return respond, express / len(states)


def injected_next_token_loss(model, reconstructed, inputs, positions, block=16):
    """Preserve the actual causal targets where concept-set losses are applied.

    Global LM CE averages over many uninjected positions. This additional local
    CE prevents a few injected states from preferring any lifted keyword over
    their correct next token. It is a retrofit-specific objective, not Eq. 32.
    """
    valid = positions[:, :-1] & inputs['attention_mask'][:, :-1].bool() & inputs['attention_mask'][:, 1:].bool()
    states = reconstructed[:, :-1][valid]
    targets = inputs['input_ids'][:, 1:][valid]
    if not len(targets):
        return reconstructed.sum() * 0
    from torch.utils.checkpoint import checkpoint
    def block_loss(h, y):
        return F.cross_entropy(model.logits(h), y, reduction='sum')
    total = states.sum() * 0
    for offset in range(0,len(states),block):
        total = total + checkpoint(block_loss,states[offset:offset+block],targets[offset:offset+block],use_reentrant=False)
    return total / len(targets)


def balanced_pools(corpus, tokenizer, lexicon, max_length, max_documents):
    """Find train chunks with actual eligible causal targets for each concept."""
    pools = {key: [] for key in lexicon}
    by_atlas = {entry['atlas_id']:(key,set(entry['tokens'])) for key,entry in lexicon.items()}
    records = corpus.splits['train'][:max_documents]
    for offset in range(0, len(records), 256):
        batch = records[offset:offset+256]
        encoded = tokenizer([row['text'] for row in batch], add_special_tokens=False)['input_ids']
        for local, (row, ids) in enumerate(zip(batch, encoded)):
            if not 2 <= len(ids) <= max_length:
                continue
            target_ids = set(ids[1:])
            for atlas_id in row['label_ids']:
                if atlas_id in by_atlas:
                    key, tokens = by_atlas[atlas_id]
                    if target_ids.intersection(tokens):
                        pools[key].append(offset+local)
    missing = [lexicon[key]['name'] for key, values in pools.items() if not values]
    if missing:
        raise ValueError('No eligible steering training chunks for: '+', '.join(missing))
    return pools
