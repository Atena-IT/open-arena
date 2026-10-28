# License Apache 2.0: (c) 2026 Atena-Reply

import synalinks

from openarena.datasets.dataset import Dataset


class HuggingFaceDataset(Dataset, synalinks.datasets.HuggingFaceDataset):
    """A HuggingFace `datasets` source: a public benchmark, or a team dataset on the Hub.

    The library's own loader, behind the engine's door: the rows are what
    `datasets.load_dataset` yields, every split in order when no `split`
    is named, and a hand-written schema is closed before it is read. Any
    other key goes to `load_dataset` (`data_files`, `token`, ...).

    Args:
        path (str): The dataset repo or builder name.
        name (str): Optional. The configuration (subset).
        split (str): Optional. The split; every split in order when None.
        revision (str): Optional. A commit, branch or tag to pin the rows to.
        streaming (bool): Download the rows on demand (the default); False
            downloads the dataset, which is also when its length is known
            without a `limit`.
        input_data_model (DataModel): See `Dataset`.
        input_schema (dict | str): See `Dataset`.
        input_template (str): See `Dataset`.
        output_data_model (DataModel): See `Dataset`.
        output_schema (dict | str): See `Dataset`.
        output_template (str): See `Dataset`.
        tools_template (str): See `Dataset`.
        batch_size (int): Examples per yielded batch. Defaults to 1.
        limit (int): Optional. See `Dataset`.
        repeat (int): See `Dataset`.
        **kwargs: Forwarded to `datasets.load_dataset`.
    """

    def __init__(
        self,
        path,
        *,
        name=None,
        split=None,
        revision=None,
        streaming=True,
        input_data_model=None,
        input_schema=None,
        input_template=None,
        output_data_model=None,
        output_schema=None,
        output_template=None,
        tools_template=None,
        batch_size=1,
        limit: int = None,
        repeat: int = 1,
        **kwargs,
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
            path=path,
            name=name,
            split=split,
            revision=revision,
            streaming=streaming,
            **kwargs,
        )
        # Images and audio come as their files' bytes, which `as_image` and `as_audio` hand on: decoding them would
        # need Pillow or torchcodec, and the model wants the file anyway.
        self._dataset = undecoded(self._dataset)


def undecoded(dataset):
    """`dataset` with its image and audio columns left undecoded: `{bytes, path}`, the file as it is stored."""
    from datasets import Audio, Image

    first = next(iter(dataset.values()), None) if hasattr(dataset, "keys") else dataset
    for name, feature in dict(getattr(first, "features", None) or {}).items():
        if isinstance(feature, (Image, Audio)) and feature.decode:
            dataset = dataset.cast_column(name, type(feature)(decode=False))
    return dataset
