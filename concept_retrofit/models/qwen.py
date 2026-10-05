"""Explicit Qwen text interface and reversible top-layer LoRA adapters."""
from contextlib import contextmanager
import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.enabled = True
        self.scale = 1.0  # alpha=rank

    def forward(self, x):
        result = self.base(x)
        if self.enabled:
            result = result + ((x.float() @ self.a.T) @ self.b.T).to(result.dtype) * self.scale
        return result


class QwenText(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.head = backbone, head
        self.requires_grad_(False)
        self.eval()
        self.adapter_names = []

    def hidden(self, inputs, all_layers=False):
        result = self.backbone(**inputs, use_cache=False, output_hidden_states=all_layers)
        return result.hidden_states if all_layers else result.last_hidden_state

    def logits(self, hidden):
        return self.head(hidden.to(self.head.weight.dtype)).float()

    def add_lora(self, top_layers, rank):
        layers = self.backbone.layers
        if not 0 < top_layers <= len(layers):
            raise ValueError(f'top_layers must be in [1,{len(layers)}]')
        for i in range(len(layers) - top_layers, len(layers)):
            for name, module in list(layers[i].named_modules()):
                if isinstance(module, nn.Linear):
                    parent_name, _, leaf = name.rpartition('.')
                    parent = layers[i].get_submodule(parent_name) if parent_name else layers[i]
                    setattr(parent, leaf, LoRALinear(module, rank))
                    self.adapter_names.append(f'layers.{i}.{name}')
        if not self.adapter_names:
            raise ValueError('No eligible top-layer linear modules')

    @contextmanager
    def teacher(self):
        adapters = [m for m in self.modules() if isinstance(m, LoRALinear)]
        previous = [m.enabled for m in adapters]
        try:
            for adapter in adapters:
                adapter.enabled = False
            with torch.no_grad():
                yield self
        finally:
            for adapter, enabled in zip(adapters, previous):
                adapter.enabled = enabled


def load_text(model_name, revision=None, device='cuda', local_files_only=False):
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
    options = {'revision': revision, 'local_files_only': local_files_only}
    config = AutoConfig.from_pretrained(model_name, **options)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **options)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device.startswith('cuda') else torch.float32
    if config.model_type == 'qwen3_5':
        wrapper = AutoModelForImageTextToText.from_pretrained(
            model_name, dtype=dtype, attn_implementation='sdpa', **options)
        # Vision parameters never occupy GPU memory for this text-only study.
        backbone, head = wrapper.model.language_model, wrapper.lm_head
    else:
        wrapper = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, attn_implementation='sdpa', **options)
        backbone, head = wrapper.model, wrapper.lm_head
    model = QwenText(backbone, head).to(device)
    return model, tokenizer
