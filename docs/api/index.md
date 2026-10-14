# API reference

openarena is a command and a Python package. Everything the command does is a
function you can call: read and check a configs folder, run a sweep, rank its
results.

```python
import asyncio
from pathlib import Path

import yaml

from openarena.config.files import evaluation_config
from openarena.evaluate import run_sweep

raw = evaluation_config(Path("my-arena/configs"), "smoke")   # the evaluation and every entry it names
run = Path("runs/smoke")
run.mkdir(parents=True, exist_ok=True)
(run / "config.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
result = asyncio.run(run_sweep(str(run / "config.yaml"), state_dir=run))
print(result["rows"])
```

| page | module | what |
|---|---|---|
| [Configuration entries](entries.md) | `openarena.config.entries` | one model per registry entry |
| [Run configuration](config.md) | `openarena.config` | a run's whole config, references resolved |
| [Checking](check.md) | `openarena.config.check` | the problems of a registry, with their fixes |
| [Running a sweep](evaluate.md) | `openarena.evaluate` | `run_sweep` |
| [Leaderboards](leaderboard.md) | `openarena.leaderboard` | gather results into a board, rank it |
| [Dataset providers](datasets.md) | `openarena.datasets` | where rows come from |
| [Rewards](rewards.md) | `openarena.rewards` | the reward catalog |
| [Harbor](harbor.md) | `openarena.harbor` | Harbor tasks, agents and sandboxes |
