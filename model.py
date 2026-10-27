"""
Build a Mini LLM Inference Server

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - stable_softmax
import numpy as np
def stable_softmax(logits):
    # TODO: compute a numerically stable softmax over the last axis of logits.
    logits = np.asarray(logits, dtype=float)
    exp_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_x/np.sum(exp_x, axis=-1, keepdims=True)

# Step 2 - apply_temperature
def apply_temperature(logits, temperature):
    # TODO: scale logits by 1 / temperature; if temperature <= 0, return logits unchanged (greedy).
    if temperature <= 0.0:
        return logits
    return logits / temperature

# Step 3 - top_k_filter
import numpy as np

def top_k_filter(logits, k):
    logits = np.asarray(logits)
    if logits.ndim not in (1, 2):
        raise ValueError("logits much be 1D or 2D")
    if not np.issubdtype(logits.dtype, np.floating):
        raise ValueError("logits must have a floating dtype to support -inf")
    if k < 0:
        raise ValueError("k must be non-negative")
    vocab_size = logits.shape[-1]
    if k >= vocab_size:
        return logits
    result = np.full_like(logits, -np.inf)
    if k == 0:
        return result
    top_values = np.partition(logits, vocab_size - k, axis=-1)[...,-k:]
    threshold = top_values.min(axis=-1, keepdims=True)
    return np.where(logits < threshold, -np.inf, logits)

# Step 4 - top_p_filter
def top_p_filter(logits, p):
    if not 0 < p <= 1:
        raise ValueError("p must be in between 0 and 1")

    logits = np.asarray(logits)
    output = logits.astype(np.result_type(logits.dtype, np.float32), copy=True)

    if p == 1:
        return output
    
    indices = np.argsort(-output, axis=-1)
    sorted_logits = np.take_along_axis(output, indices, axis=-1)

    scores = sorted_logits.astype(np.float64)
    weights = np.exp(scores - scores.max(axis=-1, keepdims= True))
    probs = weights / weights.sum(axis=-1, keepdims=True)

    cumulative = np.cumsum(probs, axis=-1)

    remove = np.zeros_like(cumulative, dtype=bool)
    remove[..., 1:] = cumulative[..., :-1] >= p

    filtered = np.where(remove, -np.inf, sorted_logits)
    np.put_along_axis(output, indices, filtered, axis=-1)

    return output

