"""Read shared datasets and record the training examples used by a run."""
from random import Random
from datasets import load_dataset


def load_split(
    repo,
    subset=None,
    split="train",
    revision=None,
    cache_dir=None,
    trust_remote_code=False,
):
    if not revision:
        raise ValueError(
            f"Dataset {repo} must use a pinned revision."
        )

    return load_dataset(
        path=repo,
        name=subset,
        split=split,
        revision=revision,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )


def sample_indices(size, count, seed):
    if not 0 < count <= size:
        raise ValueError(
            "Sample count must be between 1 and the size of the split."
        )

    return Random(seed).sample(range(size), count)