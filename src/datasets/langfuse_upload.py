import asyncio
import json
import logging
from collections import defaultdict, deque
from collections.abc import Iterable
from typing import Any

from langfuse import get_client
from tqdm.asyncio import tqdm as async_tqdm

from src.datasets.base import Row

_logger = logging.getLogger(__name__)
_MISSING_PREVIEW_LENGTH = 50
_RESERVED_METADATA_KEYS = ("lf_item_id", "lf_dataset_id", "lf_dataset_name")


def _warn_on_reserved_keys(rows: list[Row]) -> None:
    """Warn if rows already carry keys we will overwrite on upload."""
    collisions = {
        key
        for _, _, metadata in rows
        for key in _RESERVED_METADATA_KEYS
        if key in (metadata or {})
    }
    if collisions:
        _logger.warning(
            f"Row metadata contains reserved key(s) {sorted(collisions)}; "
            "these will be overwritten by Langfuse-assigned ids."
        )


def _ensure_dataset(langfuse, name: str, description: str = "") -> None:
    try:
        langfuse.get_dataset(name)
    except Exception:
        _logger.debug(f"Creating new Langfuse dataset '{name}'")
        langfuse.create_dataset(name=name, description=description)


async def upload_rows(
    rows: Iterable[Row],
    dataset_name: str,
    description: str = "",
    max_concurrency: int = 12,
) -> list[Row]:
    """
    Create/get the Langfuse dataset and upload each row. Returns new rows with
    lf_item_id / lf_dataset_id / lf_dataset_name merged into metadata so the
    execution layer can link traces back to items.
    """
    rows = list(rows)
    if not rows:
        _logger.warning("No rows to upload to Langfuse")
        return rows

    _warn_on_reserved_keys(rows)
    langfuse = get_client()
    _ensure_dataset(langfuse, dataset_name, description)

    async def _upload_one(row: Row) -> Row:
        input_, expected, metadata = row
        created = await asyncio.to_thread(
            langfuse.create_dataset_item,
            dataset_name=dataset_name,
            input=input_,
            expected_output=expected,
            metadata=metadata,
        )
        return (
            input_,
            expected,
            {
                **metadata,
                "lf_item_id": created.id,
                "lf_dataset_id": created.dataset_id,
                "lf_dataset_name": created.dataset_name,
            },
        )

    uploaded: list[Row] = []
    errors: list[str] = []
    worker_count = min(max_concurrency, len(rows))
    queue: asyncio.Queue[Row | None] = asyncio.Queue()

    for row in rows:
        queue.put_nowait(row)
    for _ in range(worker_count):
        queue.put_nowait(None)

    async def _worker() -> None:
        while True:
            row = await queue.get()
            try:
                if row is None:
                    return
                try:
                    uploaded.append(await _upload_one(row))
                except Exception as e:
                    errors.append(str(e))
                    _logger.error(f"Upload failed for row: {e}")
                finally:
                    progress.update(1)
            finally:
                queue.task_done()

    workers: list[asyncio.Task] = []
    progress = None
    try:
        progress = async_tqdm(total=len(rows), desc="Uploading to Langfuse")
        workers = [asyncio.create_task(_worker()) for _ in range(worker_count)]
        await queue.join()
        await asyncio.gather(*workers)
    finally:
        for w in workers:
            if not w.done():
                w.cancel()
        if progress is not None:
            progress.close()

    if errors:
        raise RuntimeError(
            f"{len(errors)}/{len(rows)} row uploads failed. "
            f"First error: {errors[0]}"
        )

    return uploaded


def _item_key(input_value: Any, expected_output: Any, metadata: dict[str, Any] | None) -> str:
    def normalize(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): normalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
        if isinstance(value, (list, tuple)):
            return [normalize(x) for x in value]
        return str(value)

    return json.dumps(
        {"input": normalize(input_value), "expected_output": normalize(expected_output), "metadata": normalize(metadata or {})},
        sort_keys=True,
    )


def attach_existing_dataset(rows: Iterable[Row], dataset_name: str) -> list[Row]:
    """
    For the --skip-upload path: match each row against an existing Langfuse
    dataset and annotate metadata with the discovered lf_* ids. Fails fast if
    any row cannot be matched.
    """
    rows = list(rows)
    if not rows:
        return rows

    _warn_on_reserved_keys(rows)
    langfuse = get_client()
    _logger.info(f"Loading existing Langfuse dataset: {dataset_name}")
    remote = langfuse.get_dataset(dataset_name)

    by_key: dict[str, deque] = defaultdict(deque)
    for item in remote.items:
        by_key[_item_key(item.input, item.expected_output, item.metadata)].append(item)

    annotated: list[Row] = []
    missing: list[str] = []
    for index, (input_, expected, metadata) in enumerate(rows, start=1):
        matching = by_key.get(_item_key(input_, expected, metadata))
        if not matching:
            missing.append(f"row {index} ({input_[:_MISSING_PREVIEW_LENGTH]!r})")
            annotated.append((input_, expected, metadata))
            continue
        remote_item = matching.popleft()
        annotated.append((
            input_,
            expected,
            {
                **metadata,
                "lf_item_id": remote_item.id,
                "lf_dataset_id": remote.id,
                "lf_dataset_name": remote.name,
            },
        ))

    if missing:
        preview = ", ".join(missing[:5]) + (", ..." if len(missing) > 5 else "")
        raise ValueError(
            f"Some rows were not found in existing Langfuse dataset '{dataset_name}': {preview}. "
            "Re-run without --skip-upload to upload again."
        )

    return annotated
