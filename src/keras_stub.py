# License Apache 2.0: (c) 2026 Athena-Reply

"""Stub the `keras` import surface that `keras-tuner` touches at module load.

`keras-tuner` does `import keras` and `from keras... import *` at import time
(see `keras_tuner/src/backend/{config,keras,ops,random}.py`). Real Keras 3
would then load a backend (TF/torch/jax/numpy), but we don't want that
dependency — synalinks already plays the Keras-like role for this project.
We only override `BaseTuner.run_trial`, so the keras runtime is never
actually called; we just need the import statements to succeed.

This module installs minimal stubs in `sys.modules` and must be imported
*before* `keras_tuner`.
"""

import sys
import types


def _install():
    if "keras" in sys.modules:
        return

    keras = types.ModuleType("keras")
    keras.version = lambda: "3.0.0"

    config = types.ModuleType("keras.config")
    config.backend = lambda: "numpy"
    keras.config = config

    src = types.ModuleType("keras.src")
    ops = types.ModuleType("keras.src.ops")
    src.ops = ops

    random_mod = types.ModuleType("keras.random")
    keras.random = random_mod

    callbacks = types.ModuleType("keras.callbacks")

    class Callback:
        def set_model(self, model):
            pass

    class History(Callback):
        pass

    callbacks.Callback = Callback
    callbacks.History = History
    keras.callbacks = callbacks

    utils = types.ModuleType("keras.utils")

    def serialize_keras_object(obj):
        return {
            "class_name": type(obj).__name__,
            "config": obj.get_config() if hasattr(obj, "get_config") else {},
        }

    def deserialize_keras_object(config, custom_objects=None, module_objects=None):
        if not isinstance(config, dict) or "class_name" not in config:
            return config
        class_name = config["class_name"]
        cfg = config.get("config", {})
        registry = {**(module_objects or {}), **(custom_objects or {})}
        cls = registry.get(class_name)
        if cls is None:
            raise ValueError(f"keras_stub: unknown class {class_name!r}")
        if hasattr(cls, "from_config"):
            return cls.from_config(cfg)
        return cls(**cfg)

    utils.serialize_keras_object = serialize_keras_object
    utils.deserialize_keras_object = deserialize_keras_object
    keras.utils = utils

    sys.modules["keras"] = keras
    sys.modules["keras.config"] = config
    sys.modules["keras.src"] = src
    sys.modules["keras.src.ops"] = ops
    sys.modules["keras.random"] = random_mod
    sys.modules["keras.callbacks"] = callbacks
    sys.modules["keras.utils"] = utils

    # Disable keras-tuner's image-model `applications` submodule — it imports
    # `keras.layers` and other keras-runtime symbols we don't have. We don't
    # use any HyperEfficientNet / HyperResNet / etc. anyway.
    kt_apps = types.ModuleType("keras_tuner.applications")
    kt_src_apps = types.ModuleType("keras_tuner.src.applications")
    sys.modules["keras_tuner.applications"] = kt_apps
    sys.modules["keras_tuner.src.applications"] = kt_src_apps


_install()
