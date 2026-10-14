# License Apache 2.0: (c) 2026 Atena-Reply

"""The retrieval module ranks the row's chunks by cosine with the model under test, and recall scores the gold ones."""

import asyncio
import os

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
