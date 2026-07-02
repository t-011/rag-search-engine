import numpy as np

# Buckets 0-9
NUM_BUCKETS = 10

def hash_vec(vec: np.ndarray) -> int:
    return int(sum(vec) % NUM_BUCKETS)

