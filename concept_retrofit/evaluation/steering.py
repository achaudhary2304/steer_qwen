"""Matched greedy prompt examples through the trained native bottleneck.

Semantic steering quality must be judged separately; these artifacts only
measure logit changes and preserve the actual generated text for inspection.
"""
import gc
import torch

from concept_retrofit.io import save_json


@torch.no_grad()
def generate_examples(data, checkpoint, output, max_new_tokens=32, concept_index=0):
    from concept_retrofit.pipeline import load_trained
    cfg, corpus, model, tokenizer, module = load_trained(data, checkpoint)
    generate_with_model(cfg, corpus, model, tokenizer, module, output, max_new_tokens, concept_index)
    del model, module
    gc.collect()
    if cfg.device.startswith('cuda'):
        torch.cuda.empty_cache()


@torch.no_grad()
def generate_with_model(cfg, corpus, model, tokenizer, module, output,
                        max_new_tokens=32, concept_index=0):
    """Use the resident model during training; do not reload weights or data."""
    module.eval()
    if not 0 <= concept_index < len(corpus.ids):
        raise ValueError('Invalid concept_index')
    prompts = ['Write a short story about a quiet afternoon.', 'Describe a surprising discovery in three sentences.']
    results = []
    for prompt in prompts:
        formatted = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) if tokenizer.chat_template else prompt
        encoded = tokenizer(formatted, return_tensors='pt', add_special_tokens=False).to(cfg.device)
        original_ids = encoded['input_ids']
        for condition, intervention in [('base', None), ('retrofit', None),
                                        ('amplify', {concept_index: 1.}), ('suppress', {concept_index: 0.})]:
            ids = original_ids.clone()
            for _ in range(max_new_tokens):
                inputs = {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}
                if condition == 'base':
                    with model.teacher():
                        hidden = model.hidden(inputs)
                        logits = model.logits(hidden[:, -1])
                else:
                    hidden = model.hidden(inputs)
                    reconstructed, _ = module(hidden[:, -1].float(), cfg.residual_scale, intervention)
                    logits = model.logits(reconstructed)
                next_id = logits.argmax(-1, keepdim=True)
                ids = torch.cat((ids, next_id), -1)
                if next_id.item() == tokenizer.eos_token_id:
                    break
            results.append({'prompt': prompt, 'condition': condition,
                'concept_id': corpus.ids[concept_index], 'concept_index': concept_index,
                'response': tokenizer.decode(ids[0, original_ids.shape[1]:], skip_special_tokens=True),
                'new_tokens': ids.shape[1] - original_ids.shape[1]})
    save_json(output, {'decoding': 'greedy, same prompts and token cap; no system message',
        'residual_scale': cfg.residual_scale,
        'warning': 'Inspection examples, not semantic steering success scores or activation-baseline comparisons',
        'examples': results})
