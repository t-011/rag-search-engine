import numpy as np

# Buckets 0-9
NUM_BUCKETS = 10

def hash_vec(vec: np.ndarray) -> int:
    direction = vec / np.linalg.norm(vec)
    return int(sum(direction) % NUM_BUCKETS)

