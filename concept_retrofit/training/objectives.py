"""Masked, token-normalized preservation and weak chunk supervision."""
import torch
from torch.nn import functional as F


def chunk_logits(token_logits, mask):
    # Mean logits: a length-normalized chunk classifier, NOT token annotations.
    return (token_logits * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


def annotation_loss(logits, labels, valid, positive_weight=4.0):
    if not valid.any():
        return logits.sum() * 0
    # Predict released annotation membership. Unassigned labels are proxy zeros;
    # this is deliberately not described as true semantic-negative or nnPU loss.
    return F.binary_cross_entropy_with_logits(logits[valid], labels[valid],
        pos_weight=torch.full((logits.shape[-1],), positive_weight, device=logits.device))


def language_losses(model, teacher_hidden, student_hidden, ids, mask, block=16):
    """Project small token blocks to avoid retaining sequence x vocabulary logits."""
    valid = (mask[:, :-1].bool() & mask[:, 1:].bool()).reshape(-1)
    th = teacher_hidden[:, :-1].reshape(-1, teacher_hidden.shape[-1])[valid]
    sh = student_hidden[:, :-1].reshape(-1, student_hidden.shape[-1])[valid]
    targets = ids[:, 1:].reshape(-1)[valid]
    if not len(targets):
        raise ValueError('No valid next-token targets')
    ce, kl, base_ce = sh.sum() * 0, sh.sum() * 0, sh.sum() * 0
    for start in range(0, len(targets), block):
        with torch.no_grad():
            teacher = model.logits(th[start:start+block])
            teacher_logp = teacher.log_softmax(-1)
            base_ce += F.cross_entropy(teacher, targets[start:start+block], reduction='sum')
        # Recompute head projections during backward, rather than retaining
        # many full-vocabulary softmax tensors for the whole sequence.
        from torch.utils.checkpoint import checkpoint
        def block_loss(h, logp, target):
            student = model.logits(h)
            return torch.stack((F.cross_entropy(student, target, reduction='sum'),
                F.kl_div(student.log_softmax(-1), logp, log_target=True, reduction='sum')))
        result = checkpoint(block_loss, sh[start:start+block], teacher_logp,
                            targets[start:start+block], use_reentrant=False)
        ce = ce + result[0]
        kl = kl + result[1]
    return ce / len(targets), kl / len(targets), base_ce / len(targets), len(targets)
