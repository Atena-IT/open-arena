# License Apache 2.0: (c) 2026 Atena-Reply

"""`Invalid`: a `ValueError` that says how to fix what it refuses.

Raised from the entry models' validators (`entries`, `tags`). Pydantic keeps
the exception it wraps (`error["ctx"]["error"]`), so `check` reads the fix
off it, and anything else sees a plain `ValueError` with its message.
"""

from __future__ import annotations


class Invalid(ValueError):
    def __init__(self, message: str, fix: str | None = None):
        super().__init__(message)
        self.fix = fix
