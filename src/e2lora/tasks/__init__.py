"""Loss and scoring for classification batches shared by the pilot scripts."""
import torch.nn.functional as F


def classification_loss(logits, labels):
    return F.cross_entropy(logits, labels)


def accuracy(correct, total):
    if total <= 0:
        raise ValueError("Evaluation requires at least one example.")
    return correct / total
