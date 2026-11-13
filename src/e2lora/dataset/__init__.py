"""Read shared datasets and record the training examples used by a run."""
from random import Random


def load_split(repo, subset=None, *, split, revision, cache_dir=None):
    from datasets import load_dataset

    return load_dataset(repo, subset, split=split, revision=revision, cache_dir=cache_dir)


def sample_indices(size, count, seed):
    if not 0 < count <= size:
        raise ValueError("Sample count must be between 1 and the size of the split.")
    return Random(seed).sample(range(size), count)
