from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from src.api.constants import DEFAULT_API_TOKEN


class ArenaAPIClient:
    def __init__(self, base_url: str, token: str | None = None):
        self.base_url = base_url.rstrip('/')
        self.token = token or DEFAULT_API_TOKEN

    def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        headers = {'Authorization': 'Bearer ' + self.token}
        with httpx.Client(base_url=self.base_url, headers=headers, timeout=300.0) as client:
            response = client.request(method.upper(), path, json=payload)
            response.raise_for_status()
            if response.content:
                return response.json()
            return None

    def request_file(self, method: str, path: str, file_path: str | Path | None = None) -> Any:
        payload = None
        if file_path:
            payload = json.loads(Path(file_path).read_text())
        return self.request(method, path, payload)
