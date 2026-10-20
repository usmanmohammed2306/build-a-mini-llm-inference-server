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

