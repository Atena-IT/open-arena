# License Apache 2.0: (c) 2026 Atena-Reply

"""The optional SDKs of the dataset providers, and what to install when one is missing.

The providers that read a platform (Langfuse, LangSmith, Opik, Phoenix,
Braintrust, Unity Catalog) import its SDK only when a dataset of theirs is
read, so openarena installs without them. Each is an extra of the package,
named after its provider (`openarena[langfuse]`), and `openarena[all]` has
every one.
"""

from __future__ import annotations

INSTALL = "uv tool install 'openarena[{extra}] @ git+https://github.com/Atena-IT/open-arena'"


def missing(provider: str, package: str, extra: str) -> ImportError:
    """The error to raise when a provider's SDK is not installed, saying which extra brings it."""
    return ImportError(
        f"the `{provider}` dataset provider needs the `{package}` package, which is not installed. "
        f"Install openarena with its `{extra}` extra: {INSTALL.format(extra=extra)} "
        f"(or `uv add 'openarena[{extra}]'` in a project)."
    )
