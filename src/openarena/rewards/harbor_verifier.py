# License Apache 2.0: (c) 2026 Atena-Reply

"""`harbor_verifier` — the score a Harbor task's own tests gave.

A Harbor task folder scores itself: after the agent, Harbor's verifier runs
its `tests/`, which write `/logs/verifier/reward.txt` (one number) or
`reward.json` (named numbers). The Harbor program reads them back into the
prediction (`HarborTrajectory.rewards`); this reward picks `key` there,
`reward` by default, and needs no expected answer. A trial whose verifier
wrote nothing scores 0.

```yaml
rewards:
  verifier:
    type: harbor_verifier
```
"""

from synalinks.src.rewards.reward import Reward


class HarborVerifier(Reward):
    """The verifier's score for the row: `rewards[key]` of the Harbor trajectory."""

    def __init__(self, key="reward", name="harbor_verifier", in_mask=None, out_mask=None, in_mask_pattern=None, out_mask_pattern=None):
        super().__init__(name=name)
        self.key = key
        self.in_mask = in_mask
        self.out_mask = out_mask
        self.in_mask_pattern = in_mask_pattern
        self.out_mask_pattern = out_mask_pattern

    async def call(self, y_true, y_pred):
        if y_pred is None:
            return 0.0
        rewards = (y_pred.get_json() or {}).get("rewards") or {}
        try:
            return float(rewards.get(self.key, 0.0))
        except (TypeError, ValueError):
            return 0.0

    def get_config(self):
        return {
            "key": self.key,
            "name": self.name,
            "in_mask": self.in_mask,
            "out_mask": self.out_mask,
            "in_mask_pattern": self.in_mask_pattern,
            "out_mask_pattern": self.out_mask_pattern,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
