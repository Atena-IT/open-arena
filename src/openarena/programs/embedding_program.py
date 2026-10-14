# License Apache 2.0: (c) 2026 Atena-Reply

"""Embedding program: retrieval over the row's own chunks with the embedding model under test.

Used for a dataset that declares an `embedding:` block. Each row renders, through
its input template, a `query` and the `chunks` to search; through its output
template, the gold `chunks` — the ones a good embedding model ranks first.
`EmbeddingRetriever` embeds the query and every chunk with the model the sweep
is evaluating, ranks the chunks by cosine similarity and returns the top `k`;
`retrieval_recall` then scores how many of the gold chunks were among them
(`exact_match` works too for `k = 1` and one gold chunk).

The sweep's model axis carries embedding model ids here, not LLMs.
"""

from __future__ import annotations

import math

import synalinks
from synalinks.src.backend import EmbeddingRequest, JsonDataModel, SymbolicDataModel
from synalinks.src.modules.module import Module

from openarena.programs.utils import safe_name

# What every embedding dataset's templates must render.
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "What to retrieve for."},
        "chunks": {"type": "array", "items": {"type": "string"}, "description": "The passages to search."},
    },
    "required": ["query", "chunks"],
    "additionalProperties": False,
}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "chunks": {"type": "array", "items": {"type": "string"}, "description": "The passages ranked first, best first."},
    },
    "required": ["chunks"],
    "additionalProperties": False,
}


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def rank_chunks(query_vector, chunk_vectors, chunks, k: int) -> list[str]:
    """The `k` chunks closest to the query, best first."""
    scored = sorted(zip(chunks, chunk_vectors), key=lambda cv: cosine(query_vector, cv[1]), reverse=True)
    return [chunk for chunk, _ in scored[:k]]


class EmbeddingRetriever(Module):
    """Embed the query and its chunks, return the `k` closest chunks.

    Args:
        embedding_model (EmbeddingModel | str): The model under test.
        k (int): How many chunks to return. Defaults to 1.
    """

    def __init__(self, embedding_model, k=1, **kwargs):
        super().__init__(**kwargs)
        # A bare id is built into the engine's model; anything callable (an
        # `EmbeddingModel`, or a stand-in in tests) is used as it is.
        self.embedding_model = (
            synalinks.EmbeddingModel(model=embedding_model) if isinstance(embedding_model, str) else embedding_model
        )
        self.k = int(k)
        if self.k < 1:
            raise ValueError("`k` must be at least 1")
        self.built = True

    async def compute_output_spec(self, inputs, training=False):
        return SymbolicDataModel(schema=OUTPUT_SCHEMA, name=self.name)

    async def call(self, inputs, training=False):
        if inputs is None:
            return None
        data = inputs.get_json()
        query = str(data.get("query", ""))
        chunks = [str(c) for c in (data.get("chunks") or [])]
        if not chunks:
            return JsonDataModel(json={"chunks": []}, schema=OUTPUT_SCHEMA, name=self.name)
        result = await self.embedding_model(EmbeddingRequest(texts=[query, *chunks]))
        vectors = result.get("embeddings")
        top = rank_chunks(vectors[0], vectors[1:], chunks, self.k)
        return JsonDataModel(json={"chunks": top}, schema=OUTPUT_SCHEMA, name=self.name)

    def get_config(self):
        return {"embedding_model": getattr(self.embedding_model, "model", None), "k": self.k, "name": self.name}


async def build_embedding(model_id: str, dataset, embedding_kwargs: dict, reward, metrics=None):
    """Build (and compile) the retrieval program for one embedding model.

    `embedding_kwargs` is the dataset's `embedding:` block; `k` is how many
    chunks to return (default 1). The dataset's input template must render a
    `query` and `chunks`; its output template the gold `chunks`.
    """
    inputs = synalinks.Input(schema=dataset.input_schema or INPUT_SCHEMA)
    outputs = await EmbeddingRetriever(
        embedding_model=model_id,
        k=embedding_kwargs.get("k", 1),
        name=f"retriever_{safe_name(model_id)}",
    )(inputs)
    program = synalinks.Program(inputs=inputs, outputs=outputs, name=f"embedding_{safe_name(model_id)}")
    program.compile(reward=reward, metrics=metrics or None)
    return program
