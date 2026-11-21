"""Current approved scope; archived Code scaffolding cannot schedule GPU work."""
from common import read_json

SCOPE_PATH = "experiments/chloe/configs/experiment-scope.json"


def current_scope():
    return read_json(SCOPE_PATH)


def active_tasks():
    return tuple(current_scope()["active_training_tasks"])


def require_active_task(task):
    if task not in active_tasks():
        raise ValueError(f"Task {task} is excluded from the approved Math-only scope; retained framework is inactive.")
