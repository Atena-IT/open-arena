# License Apache 2.0: (c) 2026 Atena-Reply

"""`retrieval_recall` — the share of gold chunks found among the retrieved ones.

The deterministic reward of an embedding dataset: the program returns the
`k` chunks it ranked first, the target lists the gold chunks, and the score
is `|gold ∩ retrieved| / |gold|`, 0 to 1 — recall@k. Every list-valued field
the target carries is scored this way and the fields averaged, so a target
with one `chunks` list is the common case and a richer one still works.
Chunks are compared as text, whitespace-trimmed.

```yaml
reward:
  name: retrieval_recall
```
"""

from synalinks.src.rewards.reward import Reward


def recall(gold, retrieved) -> float:
    """`|gold ∩ retrieved| / |gold|` over trimmed strings; 0 when there is no gold."""
    gold_set = {str(g).strip() for g in gold}
    if not gold_set:
        return 0.0
    hits = {str(r).strip() for r in retrieved} & gold_set
    return len(hits) / len(gold_set)


class RetrievalRecall(Reward):
    """Recall of the target's list fields in the prediction's."""

    def __init__(self, name="retrieval_recall", in_mask=None, out_mask=None, in_mask_pattern=None, out_mask_pattern=None):
        super().__init__(name=name)
        self.in_mask = in_mask
        self.out_mask = out_mask
        self.in_mask_pattern = in_mask_pattern
        self.out_mask_pattern = out_mask_pattern

    async def call(self, y_true, y_pred):
        if y_true is None or y_pred is None:
            return 0.0
        gold_json = y_true.get_json() or {}
        pred_json = y_pred.get_json() or {}
        scores = [
            recall(gold, pred_json.get(key) or [])
            for key, gold in gold_json.items()
            if isinstance(gold, list) and (not self.in_mask or key in self.in_mask) and (not self.out_mask or key not in self.out_mask)
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def get_config(self):
        return {
            "name": self.name,
            "in_mask": self.in_mask,
            "out_mask": self.out_mask,
            "in_mask_pattern": self.in_mask_pattern,
            "out_mask_pattern": self.out_mask_pattern,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
