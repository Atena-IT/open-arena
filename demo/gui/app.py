"""Entry point: launch the Open Arena GUI backend with uvicorn."""
import os
import sys
from pathlib import Path

import uvicorn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    uvicorn.run(
        "demo.gui.backend.main:app",
        host=os.getenv("OPEN_ARENA_GUI_HOST", "127.0.0.1"),
        port=int(os.getenv("OPEN_ARENA_GUI_PORT", "8000")),
        reload=os.getenv("OPEN_ARENA_GUI_RELOAD", "false").lower() in {"1", "true", "yes"},
    )


if __name__ == "__main__":
    main()
