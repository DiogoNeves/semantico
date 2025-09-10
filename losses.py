from typing import Optional

import torch
import torch.nn.functional as F


def cosine_embedding_loss_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    embed: torch.Tensor,
    temperature: float = 1.0,
    topk: Optional[int] = None,
) -> torch.Tensor:
    """
    Cosine-distance loss in the token embedding space.

    Args:
        logits: [B, T, V] unnormalized scores.
        labels: [B, T] token ids with -100 for ignored positions.
        embed:  [V, H] token embedding matrix (tied input embeddings).
        temperature: softmax temperature applied to logits before expectation.
        topk: if provided, approximate expected embedding via top-k of probs.

    Returns:
        Scalar tensor loss.
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must be [B, T, V], got {logits.shape}")
    if labels.ndim != 2:
        raise ValueError(f"labels must be [B, T], got {labels.shape}")

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    mask = shift_labels.ne(-100)
    if not mask.any():
        # No valid positions
        return logits.new_tensor(0.0)

    # Softmax with temperature
    probs = F.softmax(shift_logits / temperature, dim=-1)  # [B, T-1, V]
    E = embed.to(probs.dtype)  # [V, H]

    if topk is not None and topk > 0 and topk < probs.size(-1):
        vals, idx = torch.topk(probs, k=topk, dim=-1)  # [B, T-1, K]
        gathered = F.embedding(idx, E)  # [B, T-1, K, H]
        expected = torch.einsum("btk,btkh->bth", vals, gathered)  # [B, T-1, H]
    else:
        expected = probs @ E  # [B, T-1, H]

    # Replace ignore_index with 0 for embedding lookup; mask will zero them out
    safe_labels = shift_labels.clamp_min(0)
    target = F.embedding(safe_labels, E)  # [B, T-1, H]

    expected = F.normalize(expected, dim=-1)
    target = F.normalize(target, dim=-1)
    cos = (expected * target).sum(dim=-1)  # [B, T-1]

    loss = 1.0 - cos
    loss = (loss * mask).sum() / mask.sum()
    return loss
