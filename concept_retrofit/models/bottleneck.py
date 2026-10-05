"""A sparse named/unknown/residual bottleneck directly before a frozen LM head."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class ConceptParts:
    known_logits: Tensor
    known_activations: Tensor
    known_hidden: Tensor
    unknown_activations: Tensor
    unknown_hidden: Tensor
    residual: Tensor


def _topk_sparse(values: Tensor, k: int) -> Tensor:
    selected, indices = values.topk(k, dim=-1)
    return torch.zeros_like(values).scatter(-1, indices, selected)


class ConceptBottleneck(nn.Module):
    """Decompose final hidden states without detaching named-score gradients.

    This module belongs immediately before the existing LM head.  At a residual
    scale of one, its reconstruction is exactly the original hidden state.  A
    training schedule must lower residual reliance before claims about routing
    or preservation are made.
    """

    def __init__(self, hidden_size: int, known_concepts: int, unknown_features: int,
                 unknown_rank: int, known_topk: int, unknown_topk: int) -> None:
        super().__init__()
        if hidden_size < 1 or not 0 < unknown_rank <= min(hidden_size, unknown_features):
            raise ValueError('unknown_rank must fit the hidden and unknown feature dimensions')
        if not 0 < known_topk <= known_concepts or not 0 < unknown_topk <= unknown_features:
            raise ValueError("top-k values must fit their feature banks")
        self.known_topk, self.unknown_topk = known_topk, unknown_topk
        self.known_encoder = nn.Linear(hidden_size, known_concepts)
        self.known_vectors = nn.Parameter(torch.empty(known_concepts, hidden_size))
        self.unknown_encoder = nn.Linear(hidden_size, unknown_features)
        self.unknown_left = nn.Parameter(torch.empty(unknown_features, unknown_rank))
        self.unknown_right = nn.Parameter(torch.empty(unknown_rank, hidden_size))
        self.unknown_bias = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.known_vectors, std=0.01)
        nn.init.normal_(self.unknown_left, std=0.001)
        nn.init.normal_(self.unknown_right, std=0.001)

    def decompose(self, hidden: Tensor) -> ConceptParts:
        known_logits = self.known_encoder(hidden)
        known = _topk_sparse(known_logits.sigmoid(), self.known_topk)
        unknown = _topk_sparse(F.relu(self.unknown_encoder(hidden)), self.unknown_topk)
        known_hidden = known @ self.known_vectors
        unknown_hidden = (unknown @ self.unknown_left) @ self.unknown_right + self.unknown_bias
        residual = hidden - known_hidden - unknown_hidden
        return ConceptParts(known_logits, known, known_hidden, unknown, unknown_hidden, residual)

    def forward(self, hidden: Tensor, residual_scale: float = 1.0,
                interventions: dict[int, float] | None = None) -> tuple[Tensor, ConceptParts]:
        parts = self.decompose(hidden)
        known = parts.known_activations
        if interventions:
            known = known.clone()
            for index, value in interventions.items():
                known[..., index] = value
        known_hidden = known @ self.known_vectors
        reconstructed = known_hidden + parts.unknown_hidden + residual_scale * parts.residual
        return reconstructed, parts
