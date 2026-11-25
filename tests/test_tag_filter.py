# License Apache 2.0: (c) 2026 Atena-Reply

"""The tag filter (`openarena.config.tags`) and a leaderboard scoped by it (`dataset_tags`)."""

import json

import pytest
import yaml
from click.testing import CliRunner
from pydantic import ValidationError

from openarena.cli.main import cli, init
from openarena.config import LeaderboardEntry, TagFilter, matches_tag_filter
from openarena.config.tags import describe_tag_filter


@pytest.mark.parametrize(
    "tags, include, exclude, passes",
    [
        (["math"], [], [], True),
        (["math", "lang:it"], ["math"], [], True),
        # Two tags of one dimension: either will do.
        (["lang:en"], ["lang:it", "lang:en"], [], True),
        (["lang:fr"], ["lang:it", "lang:en"], [], False),
        # Across dimensions, and tags without one: every group must match.
        (["procurement", "lang:it"], ["procurement", "lang:it", "lang:en"], [], True),
        (["procurement"], ["procurement", "lang:it", "lang:en"], [], False),
        (["procurement", "lang:it"], ["procurement", "legal"], [], False),
        # Any excluded tag keeps an entry out.
        (["procurement", "toy"], ["procurement"], ["toy"], False),
        (["procurement"], [], ["toy"], True),
        # A tag named like a dimension is not in that dimension.
        (["lang"], ["lang:it"], [], False),
    ],
)
def test_the_filter_is_or_within_a_dimension_and_across_groups(
    tags, include, exclude, passes
):
    assert matches_tag_filter(tags, include, exclude) is passes


def test_the_filter_reads_as_one_line():
    assert (
        describe_tag_filter(["procurement", "lang:it", "lang:en"], ["toy"])
        == "procurement · lang: it or en · without toy"
    )
    assert describe_tag_filter(["lang:it"]) == "lang:it"
    assert describe_tag_filter([], ["toy", "draft"]) == "without toy or draft"


def test_a_tag_filter_is_normalized_and_names_a_tag_once():
    assert TagFilter(include=["Lang:IT", "lang:it"]).include == ["lang:it"]
    with pytest.raises(ValidationError, match="both included and excluded"):
        TagFilter(include=["toy"], exclude=["toy"])
    with pytest.raises(ValidationError, match="names no tag"):
        TagFilter()
    with pytest.raises(ValidationError):
        TagFilter(include=["a b!"])


def test_a_leaderboard_names_its_datasets_or_tags_them():
    board = LeaderboardEntry(
        title="Procurement", dataset_tags={"include": ["procurement"]}
    )
    assert board.dataset_tags.matches(["procurement"]) and board.datasets == []
    with pytest.raises(ValidationError, match="both by name"):
        LeaderboardEntry(
            title="Both", datasets=["gsm8k-20"], dataset_tags={"include": ["math"]}
        )


def test_leaderboards_run_resolves_the_tag_scope(tmp_path):
    configs, state = tmp_path / "configs", tmp_path / "state"
    init(tmp_path)
    (configs / "datasets.yaml").write_text(
        yaml.safe_dump(
            {
                "datasets": {
                    "it-tenders": {"tags": ["procurement", "lang:it"]},
                    "en-tenders": {"tags": ["procurement", "lang:en"]},
                    "toy-tenders": {"tags": ["procurement", "lang:it", "toy"]},
                    "gsm8k-20": {"tags": ["math", "lang:en"]},
                }
            }
        )
    )
    (configs / "leaderboards.yaml").write_text(
        yaml.safe_dump(
            {
                "leaderboards": {
                    "tenders": {
                        "title": "Tenders",
                        "dataset_tags": {
                            "include": ["procurement", "Lang:IT", "lang:en"],
                            "exclude": ["toy"],
                        },
                    },
                    "nothing": {
                        "title": "Nothing",
                        "dataset_tags": {"include": ["legal"]},
                    },
                }
            }
        )
    )
    run = state / "smoke"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(
        yaml.safe_dump({"models": {m: {"model": f"ollama/{m}"} for m in ("a", "b")}})
    )
    cells = {
        ("a", d): 0.9 for d in ("it-tenders", "en-tenders", "toy-tenders", "gsm8k-20")
    } | {("b", "it-tenders"): 0.5, ("b", "en-tenders"): 0.4}
    rows = [
        {"model": m, "dataset": d, "metric": "reward", "value": v, "direction": "max"}
        for (m, d), v in cells.items()
    ]
    (run / "result.json").write_text(json.dumps({"meta": {}, "rows": rows}))

    runner = CliRunner()
    args = ["--configs", str(configs), "--state-dir", str(state), "--no-eol"]
    out = tmp_path / "tenders.json"
    ok = runner.invoke(
        cli, ["leaderboards", "run", "tenders", *args, "--json", str(out)]
    )
    assert ok.exit_code == 0, ok.output
    ranking = json.loads(out.read_text())
    # `toy-tenders` is excluded and `gsm8k-20` is not procurement, so `b` is scored on every dataset in scope.
    assert sorted(ranking["datasets"]) == ["en-tenders", "it-tenders"]
    assert [r["model"] for r in ranking["rows"]] == ["a", "b"]

    none = runner.invoke(cli, ["leaderboards", "run", "nothing", *args])
    assert (
        none.exit_code == 1 and "no dataset" in none.output and "legal" in none.output
    )

    listed = runner.invoke(cli, ["leaderboards", "list", "--configs", str(configs)])
    assert "tagged procurement" in " ".join(listed.output.split())


def test_check_warns_on_a_tag_scope_no_dataset_passes():
    from openarena.config.check import check_registry

    boards = {"tenders": {"title": "Tenders", "dataset_tags": {"include": ["procurment"]}}}
    datasets = {"it-tenders": {"tags": ["procurement"]}, "odd": {"tags": "procurment"}}
    report = check_registry("leaderboards", boards, registries={"datasets": datasets})
    assert [(p.level, p.where) for p in report.problems] == [("warning", "leaderboards.tenders.dataset_tags")]
    datasets["it-tenders"]["tags"] = ["procurment"]
    assert check_registry("leaderboards", boards, registries={"datasets": datasets}).problems == []


def test_export_refuses_a_tag_scope_left_unresolved():
    from openarena.imports import export

    with pytest.raises(ValueError, match="dataset_tags"):
        export({"datasets": [], "entries": []}, {"title": "T", "dataset_tags": {"include": ["math"]}})
