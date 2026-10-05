"""Deterministic end-to-end smoke test for the causal bottleneck path.

This does not evaluate Qwen or Atlas.  It creates hidden states with known
ground-truth concept factors, trains the same bottleneck used by the project,
and verifies four mechanics: concept learning, reconstruction, teacher-logit
agreement, and a non-zero causal intervention at the LM head.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn
from torch.nn import functional as F

from concept_retrofit.models import ConceptBottleneck
from concept_retrofit.training.losses import teacher_kl


@dataclass
class SmokeReport:
    seed: int
    steps: int
    known_auc: float
    reconstruction_mse: float
    teacher_kl: float
    intervention_logit_shift: float
    passed: bool


def _make_fixture(seed: int, examples: int, hidden_size: int, known: int,
                  unknown_rank: int, vocabulary: int) -> tuple[Tensor, Tensor, nn.Linear]:
    generator = torch.Generator().manual_seed(seed)
    # Exactly one active known factor makes the expected concept label and
    # intervention unambiguous. Unknown factors carry independent detail.
    labels = F.one_hot(torch.randint(known, (examples,), generator=generator), known).float()
    known_basis = F.normalize(torch.randn(known, hidden_size, generator=generator), dim=-1) * 3.5
    unknown_codes = torch.randn(examples, unknown_rank, generator=generator)
    unknown_basis = F.normalize(torch.randn(unknown_rank, hidden_size, generator=generator), dim=-1) * 1.2
    noise = torch.randn(examples, hidden_size, generator=generator) * 0.03
    hidden = labels @ known_basis + unknown_codes @ unknown_basis + noise
    head = nn.Linear(hidden_size, vocabulary, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.randn(vocabulary, hidden_size, generator=generator) / hidden_size**0.5)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return hidden, labels, head


def run(seed: int = 1729, steps: int = 500, device: str = "cpu") -> SmokeReport:
    torch.manual_seed(seed)
    hidden_size, known, unknown, rank, vocabulary = 32, 6, 24, 8, 48
    hidden, labels, head = _make_fixture(seed, 768, hidden_size, known, rank, vocabulary)
    hidden, labels, head = hidden.to(device), labels.to(device), head.to(device)
    train_hidden, train_labels = hidden[:600], labels[:600]
    test_hidden, test_labels = hidden[600:], labels[600:]
    module = ConceptBottleneck(hidden_size, known, unknown, rank, known_topk=2, unknown_topk=4).to(device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=3e-3, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    module.train()
    for step in range(steps):
        index = torch.randint(len(train_hidden), (96,), generator=generator, device=device)
        h, y = train_hidden[index], train_labels[index]
        # The residual starts permissive, then forces K + U to carry more of h.
        residual_scale = 1.0 - 0.75 * min(1.0, step / max(1, int(steps * 0.6)))
        reconstructed, parts = module(h, residual_scale=residual_scale)
        student, teacher = head(reconstructed), head(h).detach()
        concept = F.binary_cross_entropy_with_logits(parts.known_logits, y)
        reconstruction = F.mse_loss(reconstructed, h)
        loss = 2.0 * concept + reconstruction + 0.25 * teacher_kl(student, teacher)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    module.eval()
    with torch.no_grad():
        hidden, labels = test_hidden, test_labels
        reconstructed, parts = module(hidden, residual_scale=0.25)
        auc = roc_auc_score(labels.cpu().numpy().ravel(), parts.known_logits.cpu().numpy().ravel())
        mse = F.mse_loss(reconstructed, hidden).item()
        kl = teacher_kl(head(reconstructed), head(hidden)).item()
        target = 0
        selected = torch.where(labels[:, target] == 0)[0]
        baseline, _ = module(hidden[selected], residual_scale=0.25)
        intervened, _ = module(hidden[selected], residual_scale=0.25, interventions={target: 1.0})
        shift = (head(intervened) - head(baseline)).abs().mean().item()
    # Thresholds intentionally test only plumbing, not model quality.
    passed = auc > 0.90 and mse < 0.30 and kl < 0.15 and shift > 1e-4
    return SmokeReport(seed, steps, auc, mse, kl, shift, passed)


def save_report(report: SmokeReport, output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2) + "\n")
