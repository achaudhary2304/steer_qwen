"""Losses used by the planned sequence-level post-training objective."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def teacher_kl(student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
    """Keep the retrofitted model near the untouched Qwen teacher."""
    return F.kl_div(F.log_softmax(student_logits, -1), F.log_softmax(teacher_logits, -1),
                    log_target=True, reduction="batchmean")


def positive_unlabeled_concept_loss(logits: Tensor, positive_mask: Tensor) -> Tensor:
    """Positive-only concept loss; unlabeled concepts are not treated as negatives."""
    if positive_mask.dtype is not torch.bool:
        raise TypeError("positive_mask must be boolean")
    if not positive_mask.any():
        return logits.sum() * 0
    return F.binary_cross_entropy_with_logits(logits[positive_mask], torch.ones_like(logits[positive_mask]))


def residual_energy(residual: Tensor) -> Tensor:
    return residual.square().mean()
