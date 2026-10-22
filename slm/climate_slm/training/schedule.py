"""Learning-rate schedule and batching order (pure Python)."""
import math
import random


def steps_per_epoch(n_rows: int, effective_batch: int) -> int:
    """Optimiser steps per epoch. The last step of an epoch may use fewer rows."""
    return math.ceil(n_rows / effective_batch)


def warmup_steps(total_steps: int, warmup_ratio: float) -> int:
    return max(1, math.ceil(total_steps * warmup_ratio))


def lr_at_step(step: int, total_steps: int, base_lr: float, warmup_ratio: float) -> float:
    """Learning rate for the optimiser step with 0-based index `step`.
    Linear warm-up (step 0 already has lr * 1/W, so no step is wasted), then cosine decay towards 0."""
    w = warmup_steps(total_steps, warmup_ratio)
    if step < w:
        return base_lr * (step + 1) / w
    progress = (step - w) / max(1, total_steps - w)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def epoch_order(n_rows: int, seed: int, epoch: int) -> list[int]:
    """Row order of one epoch (1-based epoch). Fixed by the seed, so the data order is pinned."""
    order = list(range(n_rows))
    random.Random(seed * 1000 + epoch).shuffle(order)
    return order


def optimizer_steps(order: list[int], effective_batch: int, micro_batch: int) -> list[list[list[int]]]:
    """Split one epoch into optimiser steps; each step is a list of micro-batches of row indices."""
    steps = []
    for i in range(0, len(order), effective_batch):
        group = order[i:i + effective_batch]
        steps.append([group[j:j + micro_batch] for j in range(0, len(group), micro_batch)])
    return steps


def repeat_refusals(rows: list[dict], repeat: int) -> list[dict]:
    """Every refusal row appears `repeat` times (1 = unchanged). Order of the rows is kept, copies follow the original."""
    if repeat < 1:
        raise ValueError("refusal_repeat must be at least 1")
    out = []
    for r in rows:
        out.extend([r] * (repeat if r["meta"]["type"] == "unanswerable" else 1))
    return out
