import random
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn


def _optimizer_and_scheduler(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / 3)
    )
    return optimizer, scheduler


def _step(model, optimizer, scheduler, batch):
    features, labels = batch
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model(features), labels)
    loss.backward()
    optimizer.step()
    scheduler.step()


class CheckpointResumeTests(unittest.TestCase):
    def test_optimizer_scheduler_and_rng_resume_exactly(self):
        batches = [
            (torch.randn(8, 4), torch.randint(0, 2, (8,)))
            for _ in range(4)
        ]

        torch.manual_seed(2026)
        random.seed(2026)
        initial = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.25), nn.Linear(4, 2))
        initial_state = {name: value.clone() for name, value in initial.state_dict().items()}

        reference = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.25), nn.Linear(4, 2))
        reference.load_state_dict(initial_state)
        ref_optimizer, ref_scheduler = _optimizer_and_scheduler(reference)
        torch.manual_seed(77)
        random.seed(77)
        for batch in batches:
            _step(reference, ref_optimizer, ref_scheduler, batch)

        interrupted = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.25), nn.Linear(4, 2))
        interrupted.load_state_dict(initial_state)
        optimizer, scheduler = _optimizer_and_scheduler(interrupted)
        torch.manual_seed(77)
        random.seed(77)
        for batch in batches[:2]:
            _step(interrupted, optimizer, scheduler, batch)

        checkpoint = {
            "model": interrupted.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "global_step": 2,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.pt"
            torch.save(checkpoint, path)
            saved = torch.load(path, map_location="cpu", weights_only=True)

        resumed = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.25), nn.Linear(4, 2))
        resumed.load_state_dict(saved["model"])
        resumed_optimizer, resumed_scheduler = _optimizer_and_scheduler(resumed)
        resumed_optimizer.load_state_dict(saved["optimizer"])
        resumed_scheduler.load_state_dict(saved["scheduler"])
        torch.set_rng_state(saved["torch_rng"])
        random.setstate(tuple(saved["python_rng"]))
        for batch in batches[2:]:
            _step(resumed, resumed_optimizer, resumed_scheduler, batch)

        for name, value in reference.state_dict().items():
            self.assertTrue(torch.equal(value, resumed.state_dict()[name]), name)
        self.assertEqual(ref_scheduler.state_dict(), resumed_scheduler.state_dict())
        self.assertEqual(ref_optimizer.state_dict()["param_groups"], resumed_optimizer.state_dict()["param_groups"])


if __name__ == "__main__":
    unittest.main()
