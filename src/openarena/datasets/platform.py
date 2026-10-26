# License Apache 2.0: (c) 2026 Atena-Reply

"""Datasets read from an evaluation platform through its SDK.

Langfuse, LangSmith, Braintrust, Opik, Phoenix and Unity Catalog share this
base. Each names the SDK it reads with, imported only when a dataset of its
kind is opened so openarena installs without it, and answers three
questions: how to reach the platform (`connect`, done once, when the dataset
is built, so a wrong name fails the run before any trial), which records
the dataset holds (`records`) and what row each one is to the templates
(`row`), and how many there are when the platform can say without reading
them all (`count`). The base turns those into the engine's `_iter_rows` and
`__len__`, keeps the provider's settings as attributes, and says which extra
installs a missing SDK.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from openarena.datasets.dataset import Dataset
from openarena.datasets.extras import missing


class PlatformDataset(Dataset):
    """A dataset held by an evaluation platform, read through its SDK."""

    # The registry name, the module the SDK is imported as, the package that ships it, the extra that installs it.
    provider = ""
    sdk = ""
    package = ""
    extra = ""

    def __init__(
        self,
        *,
        input_data_model=None,
        input_schema=None,
        input_template=None,
        output_data_model=None,
        output_schema=None,
        output_template=None,
        tools_template=None,
        batch_size=1,
        limit: int | None = None,
        repeat: int = 1,
        **settings: Any,
    ):
        super().__init__(
            input_data_model=input_data_model,
            input_schema=input_schema,
            input_template=input_template,
            output_data_model=output_data_model,
            output_schema=output_schema,
            output_template=output_template,
            tools_template=tools_template,
            batch_size=batch_size,
            limit=limit,
            repeat=repeat,
        )
        for key, value in settings.items():
            setattr(self, key, value)
        self.client = self.connect()

    # -- what each platform answers ----------------------------------------------

    def connect(self) -> Any:
        """Reach the platform, with the settings given: the handle `records` and `count` read from."""
        raise NotImplementedError

    def records(self) -> Iterable[Any]:
        """The dataset's records, as the SDK hands them over, in the platform's order."""
        raise NotImplementedError

    def row(self, record: Any) -> dict:
        """What a record is to the templates: a mapping of the names a template may use."""
        return record

    def count(self) -> int | None:
        """How many records there are, when the platform can say without reading them all."""
        return None

    # -- the engine's side ---------------------------------------------------------

    def load_sdk(self):
        """The platform's SDK module, or the error saying which extra of openarena brings it."""
        try:
            return importlib.import_module(self.sdk)
        except ImportError as e:
            raise missing(self.provider, self.package, self.extra) from e

    def _iter_rows(self):
        for record in self.records():
            yield self.row(record)

    def __len__(self):
        n = self.limit if self.limit is not None else self.count()
        if n is None:
            raise NotImplementedError(
                f"a {self.provider} dataset has an unknown length; set `limit` for a bounded epoch."
            )
        return self._total_batches(n)


def utc(moment: str | datetime | None) -> datetime | None:
    """A point in time as a timezone-aware datetime: ISO 8601 text (`Z` accepted), a datetime read as UTC when naive, or None."""
    if moment is None:
        return None
    if not isinstance(moment, datetime):
        moment = datetime.fromisoformat(str(moment))
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
