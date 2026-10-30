# License Apache 2.0: (c) 2026 Atena-Reply

"""Harbor task folders as a dataset: one row per task, run as Harbor runs it.

A Harbor task is a folder (`task.toml`, `instruction.md`, `environment/`,
`tests/`, `solution/`, maybe `steps/`): what the agent is told, the sandbox it
works in and the tests that score it all live there. This provider reads
the folders under `path` (each one holding a `task.toml`), or only those
`tasks` names, sorted, and yields each as `{task, path, question}`: its
name, its folder and its instruction, the question the trials' log and the
answers' view show. The Harbor program runs the folder itself; nothing of it
is templated, so the templates are fixed here; the target only names the
task, since its tests score it.
"""

from __future__ import annotations

from pathlib import Path

from openarena.datasets.dataset import Dataset
from openarena.harbor import task_clock

INPUT_TEMPLATE = '{"task": {{ task | tojson }}, "path": {{ path | tojson }}, "question": {{ question | tojson }}}'
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "task": {"type": "string", "description": "The task's folder name"},
        "path": {"type": "string", "description": "The task's folder"},
        "question": {"type": "string", "description": "The task's instruction.md"},
    },
    "required": ["task", "path", "question"],
    "additionalProperties": False,
}
# The target: the task's own tests score it (`harbor_verifier`), so the target only names the task; a reward is
# still compiled against targets, and each row's target is what samples and votes index it by.
OUTPUT_TEMPLATE = '{"task": {{ task | tojson }}}'
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"task": {"type": "string", "description": "The task's folder name"}},
    "required": ["task"],
    "additionalProperties": False,
}


def task_folders(path: str | Path, tasks: list[str] | None = None) -> list[Path]:
    """The task folders under `path`: those named in `tasks`, else every folder holding a `task.toml`, sorted by name."""
    root = Path(path)
    if tasks:
        missing = [t for t in tasks if not (root / t / "task.toml").is_file()]
        if missing:
            raise FileNotFoundError(f"no Harbor task {', '.join(map(repr, missing))} under {root} (a task is a folder with a task.toml)")
        return [root / t for t in tasks]
    return sorted((p for p in root.iterdir() if (p / "task.toml").is_file()), key=lambda p: p.name) if root.is_dir() else []


def instruction_of(folder: Path) -> str:
    """What the task tells the agent: its `instruction.md`, or its first step's for a multi-step task."""
    own = folder / "instruction.md"
    if own.is_file():
        return own.read_text(errors="replace").strip()
    steps = sorted((folder / "steps").glob("*/instruction.md")) if (folder / "steps").is_dir() else []
    return steps[0].read_text(errors="replace").strip() if steps else ""


class HarborTasksDataset(Dataset):
    """Harbor task folders, one row each.

    Args:
        path (str): The folder holding the tasks, one sub-folder each.
        tasks (list): Optional. The tasks to run, by folder name; none runs every task under `path`.
        batch_size (int): Tasks run at once.
        limit (int): Optional. At most this many tasks.
        repeat (int): Optional. Each task this many times.
    """

    def __init__(self, path, *, tasks=None, batch_size=1, limit: int = None, repeat: int = 1, **ignored):
        super().__init__(
            input_schema=INPUT_SCHEMA,
            input_template=INPUT_TEMPLATE,
            output_schema=OUTPUT_SCHEMA,
            output_template=OUTPUT_TEMPLATE,
            batch_size=batch_size,
            limit=limit,
            repeat=repeat,
        )
        self.path = Path(path)
        self.tasks = list(tasks) if tasks else None
        if not self.path.is_dir():
            raise FileNotFoundError(f"no Harbor tasks folder at {self.path}")
        # A clock a task declares wrongly stops the dataset here, naming the task, before any trial.
        for folder in task_folders(self.path, self.tasks):
            task_clock(folder)

    def __len__(self):
        n = len(task_folders(self.path, self.tasks))
        return self._total_batches(min(n, self.limit) if self.limit is not None else n)

    def _iter_rows(self):
        for folder in task_folders(self.path, self.tasks):
            yield {"task": folder.name, "path": str(folder.resolve()), "question": instruction_of(folder)}
