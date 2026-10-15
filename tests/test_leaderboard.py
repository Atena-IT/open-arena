# License Apache 2.0: (c) 2026 Atena-Reply

"""`openarena.leaderboard`: the board gathered from sweep results, and the ranking the leaderboard page shows."""

from datetime import date

import pytest

from openarena import leaderboard as lb


def _result(cells):
    """A `run_sweep` result: `cells` is `{(model, dataset): {metric: value}}`."""
    rows = [
        {"model": m, "dataset": d, "metric": k, "value": v, "direction": "min" if k == "cost" else "max"}
        for (m, d), metrics in cells.items()
        for k, v in metrics.items()
    ]
    return {"meta": {}, "rows": rows}


def test_rank_weighs_and_marks_the_frontier():
    runs = [
        {
            "id": "r",
            "result": _result(
                {
                    ("cheap", "d"): {"reward": 0.5, "cost": 0.01},
                    ("good", "d"): {"reward": 0.9, "cost": 1.0},
                    ("bad", "d"): {"reward": 0.4, "cost": 2.0},
                    ("partial", "d"): {"reward": 1.0},
                }
            ),
        }
    ]
    data = lb.board(runs, eol=False)
    # `partial` lacks cost, so cost cannot carry weight in a scope holding it: reward alone decides.
    everyone = lb.rank(data, shares={"reward": 50, "cost": 50})
    assert everyone["shares"] == {"reward": 100.0}
    assert everyone["rows"][0]["model"] == "partial"

    three = lb.rank(data, models=["cheap", "good", "bad"], shares={"reward": 50, "cost": 50})
    assert three["shares"] == {"cost": 50.0, "reward": 50.0}
    assert [r["model"] for r in three["rows"]] == ["cheap", "good", "bad"]
    assert {r["model"] for r in three["rows"] if r["on_frontier"]} == {"cheap", "good"}

    # No shares: the default preset, reward high and cost low.
    default = lb.rank(data, models=["cheap", "good", "bad"])
    assert default["shares"] == pytest.approx({"reward": 90.0, "cost": 10.0})
    assert default["rows"][0]["model"] == "good"


def test_board_takes_each_cell_whole_from_the_newest_run():
    runs = [  # newest first
        {"id": "new", "result": _result({("a", "d"): {"reward": 0.9}}), "model_ids": {"a": "openai/gpt-4o"}},
        {
            "id": "old",
            "result": _result(
                {
                    ("a", "d"): {"reward": 0.1, "cost": 5.0},  # replaced whole: its cost does not leak into the new cell
                    ("a", "e"): {"reward": 0.3},
                    ("b", "d"): {"cost": 1.0},  # no reward: not a cell
                }
            ),
        },
    ]
    data = lb.board(runs, eol=False)
    assert data["datasets"] == ["d", "e"]
    assert data["metrics"] == ["reward"]  # only what the cells kept recorded
    (a,) = data["entries"]
    assert a["model_id"] == "openai/gpt-4o"
    assert a["scores"]["d"]["run_id"] == "new"
    assert set(a["scores"]["d"]["metrics"]) == {"reward"}
    assert a["scores"]["e"]["run_id"] == "old"
    assert a["mean"] == pytest.approx(0.6)


def test_board_adds_time_to_end_of_life_to_every_cell():
    runs = [{"id": "r", "result": _result({("m", "d"): {"reward": 1.0}, ("m", "e"): {"reward": 0.5}}), "model_ids": {"m": "ollama/tiny"}}]
    data = lb.board(runs, today=date(2026, 1, 1))
    assert data["metrics"] == ["reward", "eol_days"]
    days = {cell["metrics"]["eol_days"]["value"] for cell in data["entries"][0]["scores"].values()}
    assert len(days) == 1  # a fact about the model, the same on every dataset


def test_normalize_is_linear_for_the_reward_and_log_for_magnitudes():
    assert lb.normalize("reward", "max", [0.0, 0.5, 1.0]) == pytest.approx([0.05, 0.525, 1.0])
    # Cost, lower better, on a log scale: 10x cheaper counts as much as the next 10x.
    assert lb.normalize("cost", "min", [0.01, 0.1, 1.0]) == pytest.approx([1.0, 0.525, 0.05])
    assert lb.normalize("reward", "max", [0.7, 0.7]) == [1.0, 1.0]


def test_a_model_must_be_scored_on_every_dataset_in_scope():
    runs = [{"id": "r", "result": _result({("a", "d"): {"reward": 1.0}, ("a", "e"): {"reward": 1.0}, ("b", "d"): {"reward": 0.5}})}]
    data = lb.board(runs, eol=False)
    both = lb.rank(data)
    assert [r["model"] for r in both["rows"]] == ["a"]
    assert both["skipped"] == ["b"] and both["gaps"]["b"] == [{"dataset": "e", "missing": None}]
    # Narrowed to `d`, `b` is back in the field.
    assert [r["model"] for r in lb.rank(data, datasets=["d"])["rows"]] == ["a", "b"]


def test_a_budget_leaves_out_the_models_that_break_it_before_normalizing():
    cells = {("cheap", "d"): {"reward": 0.5, "cost": 0.01}, ("good", "d"): {"reward": 0.9, "cost": 1.0}, ("pricey", "d"): {"reward": 0.95, "cost": 9.0}}
    data = lb.board([{"id": "r", "result": _result(cells)}], eol=False)
    ranked = lb.rank(data, shares={"reward": 100}, budgets=[{"metric": "cost", "op": "lte", "value": 2.0}])
    assert [r["model"] for r in ranked["rows"]] == ["good", "cheap"]
    assert ranked["excluded"] == [{"model": "pricey", "failed": [{"metric": "cost", "op": "lte", "value": 2.0}]}]
    assert "pricey" not in ranked["skipped"]
    # Without `pricey` stretching it, the reward scale runs from `cheap` to `good`.
    assert ranked["rows"][0]["contributions"][0]["normalized"] == 1.0
    assert lb.violates({"metric": "reward", "op": "gte", "value": 0.6}, 0.5) and not lb.violates({"metric": "x", "op": "gte", "value": 1}, None)
# --- weights per group and per dataset ------------------------------------------------


def _scores(cells: dict[tuple[str, str], float], groups: dict[str, str] | None = None) -> dict:
    rows = [{"model": m, "dataset": d, "metric": "reward", "value": v, "direction": "max"} for (m, d), v in cells.items()]
    data = lb.board([{"id": "import:x", "result": {"rows": rows}, "groups": groups or {}}], eol=False)
    return data


def test_a_group_weighs_its_share_whatever_its_size():
    w = lb.dataset_weights(["a1", "a2", "b"], {"a1": "a", "a2": "a", "b": "b"})
    assert w == {"a1": 0.5, "a2": 0.5, "b": 1.0}  # two datasets share group a's 1, b alone holds 1
    w = lb.dataset_weights(["a1", "a2", "b"], {"a1": "a", "a2": "a", "b": "b"}, {"a": 2}, {"a1": 3})
    assert w == {"a1": 1.5, "a2": 0.5, "b": 1.0}


def test_no_weights_is_the_plain_mean_and_weights_move_the_ranking():
    # `n` wins `x`, `m` wins `y`: on a tie `m` sorts first, so only the weight can put `n` on top.
    cells = {("m", "x"): 0.0, ("m", "y"): 1.0, ("n", "x"): 1.0, ("n", "y"): 0.0, ("o", "x"): 0.5, ("o", "y"): 0.5}
    data = _scores(cells)
    plain = {r["model"]: r["score"] for r in lb.rank(data, shares={"reward": 100})["rows"]}
    assert plain["m"] == pytest.approx(plain["n"])  # a tie on the plain mean
    for weighed in ({"group_shares": {"x": 3}}, {"dataset_shares": {"x": 3}}):
        tilted = lb.rank(data, shares={"reward": 100}, **weighed)["rows"]
        assert [r["model"] for r in tilted] == ["n", "o", "m"], weighed
        top = tilted[0]
        assert top["score"] == pytest.approx((3 * 1.0 + 0.05) / 4)
        assert sum(top["dataset_contributions"].values()) == pytest.approx(top["score"])  # the score, decomposed


def test_dataset_shares_weigh_datasets_in_no_group():
    # A CLI run names no groups: each dataset is its own, and its `dataset_shares` is its weight.
    assert lb.dataset_weights(["x", "y"], None, None, {"x": 3}) == {"x": 3.0, "y": 1.0}
    assert lb.dataset_weights(["x", "y"], None, {"x": 2}, {"x": 3, "y": 0}) == {"x": 6.0, "y": 0.0}
    # Next to a real group, an ungrouped dataset still counts as a group of one.
    w = lb.dataset_weights(["a1", "a2", "b"], {"a1": "a", "a2": "a"}, None, {"a1": 3, "b": 2})
    assert w == {"a1": 0.75, "a2": 0.25, "b": 2.0}


def test_dataset_shares_split_a_named_group_as_before():
    # An import's groups, even a group of one, split their share among their datasets.
    groups = {"a1": "a", "a2": "a", "b": "b"}
    assert lb.dataset_weights(["a1", "a2", "b"], groups, None, {"a1": 3, "b": 5}) == {"a1": 0.75, "a2": 0.25, "b": 1.0}
    cells = {("m", "a1"): 1.0, ("m", "a2"): 0.0, ("m", "b"): 0.0, ("n", "a1"): 0.0, ("n", "a2"): 1.0, ("n", "b"): 1.0}
    data = _scores(cells, groups)
    ranked = lb.rank(data, shares={"reward": 100}, dataset_shares={"a1": 3, "b": 5})["rows"]
    # m: (0.75 * 1 + 0.25 * 0.05 + 1 * 0.05) / 2; n: (0.75 * 0.05 + 0.25 * 1 + 1 * 1) / 2
    assert {r["model"]: r["score"] for r in ranked} == pytest.approx({"m": 0.40625, "n": 0.64375})


def test_a_dataset_in_no_group_does_not_share_a_group_named_like_it():
    # `a` stands alone; the group `a` holds `a1` only, which keeps its whole share.
    assert lb.dataset_weights(["a1", "a"], {"a1": "a"}) == {"a1": 1.0, "a": 1.0}


def test_a_dataset_weighed_zero_is_neither_scored_nor_required():
    data = _scores({("m", "x"): 1.0, ("m", "diag"): 0.2, ("n", "x"): 0.4}, {"x": "main", "diag": "diagnostic"})
    assert [r["model"] for r in lb.rank(data, shares={"reward": 100})["rows"]] == ["m"]  # n lacks diag
    ranked = lb.rank(data, shares={"reward": 100}, group_shares={"diagnostic": 0})["rows"]
    assert [r["model"] for r in ranked] == ["m", "n"]
    assert data["groups"] == {"x": "main", "diag": "diagnostic"}
