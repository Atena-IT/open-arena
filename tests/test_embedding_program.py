# License Apache 2.0: (c) 2026 Atena-Reply

"""The retrieval module ranks the row's chunks by cosine with the model under test, and recall scores the gold ones."""

import asyncio
import os

import pytest

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from synalinks.src.backend import JsonDataModel

from openarena.programs.embedding_program import INPUT_SCHEMA, EmbeddingRetriever, rank_chunks
from openarena.rewards.retrieval_recall import RetrievalRecall, recall

# A toy embedding: one axis per vocabulary word, counting occurrences.
VOCAB = ["cats", "dogs", "sleep", "sleeping", "bark", "run"]


def _embed(text: str):
    words = text.lower().split()
    return [float(words.count(v)) for v in VOCAB]


class FakeEmbeddingModel:
    model = "fake/embed"

    async def __call__(self, request):
        texts = request.get("texts")
        return JsonDataModel(json={"embeddings": [_embed(t) for t in texts]}, schema={"type": "object", "properties": {"embeddings": {"type": "array"}}})


def test_rank_chunks_puts_the_closest_first():
    vectors = [_embed("cats sleep"), _embed("dogs bark"), _embed("sleeping cats")]
    assert rank_chunks(_embed("sleeping cats"), vectors, ["cats sleep", "dogs bark", "sleeping cats"], 2) == ["sleeping cats", "cats sleep"]


def test_the_module_returns_the_top_k_chunks_for_the_query():
    module = EmbeddingRetriever(embedding_model=FakeEmbeddingModel(), k=1)
    row = JsonDataModel(json={"query": "where do cats sleep", "chunks": ["dogs bark at night", "cats sleep on sofas", "people run"]}, schema=INPUT_SCHEMA)
    assert asyncio.run(module(row)).get_json() == {"chunks": ["cats sleep on sofas"]}


def test_recall_over_the_gold_chunks():
    assert recall(["a", "b"], ["b", "c"]) == 0.5
    assert recall([], ["a"]) == 0.0
    reward = RetrievalRecall()
    gold = JsonDataModel(json={"chunks": ["a", "b"]}, schema={"type": "object", "properties": {"chunks": {"type": "array"}}})
    pred = JsonDataModel(json={"chunks": [" a ", "z"]}, schema={"type": "object", "properties": {"chunks": {"type": "array"}}})
    assert asyncio.run(reward(gold, pred)) == 0.5


def test_the_model_under_test_is_built_with_its_dimensions():
    from openarena.programs.embedding_program import build_embedding
    from openarena.rewards import get

    class Rows:
        input_schema = None

    program = asyncio.run(build_embedding("openai/text-embedding-3-small", Rows(), {"k": 2}, get("retrieval_recall"), lm_kwargs={"dimensions": 256}))
    [retriever] = [m for m in program._flatten_modules(recursive=True) if isinstance(m, EmbeddingRetriever)]
    assert retriever.embedding_model.model == "openai/text-embedding-3-small"
    assert retriever.embedding_model.default_kwargs == {"dimensions": 256}


def test_a_sweep_hands_the_model_under_test_its_dimensions(tmp_path, monkeypatch):
    import json

    import yaml

    import openarena.evaluate as evaluate

    seen = {}

    class Stop(Exception):
        pass

    async def build(model_id, dataset, embedding_kwargs, reward, metrics=None, lm_kwargs=None):
        seen.update(model=model_id, lm_kwargs=lm_kwargs)
        raise Stop

    monkeypatch.setattr(evaluate, "build_embedding", build)
    (tmp_path / "rows.jsonl").write_text(json.dumps({"query": "q", "passages": ["a", "b"], "gold": ["a"]}) + "\n")
    config = {
        "models": {"minilm": {"model": "ollama/all-minilm", "type": "embedding", "params": {"dimensions": 384}}},
        "rewards": {"recall": {"type": "retrieval_recall"}},
        "datasets": {
            "passages": {
                "type": "local",
                "config": {
                    "path": str(tmp_path / "rows.jsonl"),
                    "input_template": '{"query": {{ query | tojson }}, "chunks": {{ passages | tojson }}}',
                    "output_template": '{"chunks": {{ gold | tojson }}}',
                },
                "embedding": {"k": 1},
                "reward": "recall",
            }
        },
        "evaluation": {"name": "e", "language_models": ["minilm"], "datasets": ["passages"]},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    with pytest.raises(Stop):
        asyncio.run(evaluate.run_sweep(str(tmp_path / "config.yaml"), state_dir=tmp_path / "state"))
    assert seen == {"model": "ollama/all-minilm", "lm_kwargs": {"dimensions": 384}}
