# License Apache 2.0: (c) 2026 Atena-Reply

"""Harbor's installed agents, each writing down the version it runs once installed, pinned or not.

Harbor 0.24 runs an agent's version command after its install only when no version was asked for: with a pin, the
trial's `agent_info.version` is the pin, whatever ran. Each class here runs that command after every install and writes
what it says to `version.txt` in the agent's logs, which the trial's line reads (`version_line`). `run_trial` runs these
classes for the built-in agents Harbor installs. Remove once Harbor checks a pinned install.
"""

import re
from pathlib import Path
from typing import Any

from harbor.agents.factory import AgentFactory
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.models.agent.name import AgentName

from openarena.config.entries import HARBOR_AGENTS, LATEST
from openarena.harbor_hermes import Hermes

FILE = "version.txt"


class Detects:
    """Mixed in before a Harbor installed agent: after its setup, what its version command says goes to `FILE`."""

    async def setup(self, environment) -> None:
        await super().setup(environment)
        command = self.get_version_command()
        if not command:
            return
        try:
            said = await environment.exec(command=command)
            # Some agents print warnings before it (OpenHands: grpc's FutureWarning): the last line is the version.
            if said.return_code == 0 and (lines := self.parse_version(said.stdout or "").strip().splitlines()):
                (Path(self.logs_dir) / FILE).write_text(lines[-1])
        except Exception:  # noqa: BLE001 - best effort, as Harbor's own: a trial never fails on reading a version
            pass


def _attr(agent: str) -> str:
    return re.sub(r"\W", "_", agent)


# One class per built-in agent Harbor installs, named after it (`claude_code`); Hermes is the engine's own.
CLASSES = {
    agent: cls
    for agent in HARBOR_AGENTS
    if issubclass(base := Hermes if agent == "hermes" else AgentFactory.get_agent_class(AgentName(agent)), BaseInstalledAgent)
    for cls in [type(base.__name__, (Detects, base), {"__module__": __name__})]
}
globals().update({_attr(agent): cls for agent, cls in CLASSES.items()})


def agent_path(agent: str) -> str:
    """What a trial names for `agent`: the class here for a built-in agent Harbor installs, else `agent` as written."""
    return f"{__name__}:{_attr(agent)}" if agent in CLASSES else agent


def version_line(requested: str | None, origin: str | None, agent_dir: Path) -> dict[str, Any]:
    """A trial's `agent_version`: the version asked for, the one that ran (None when unread) and where the ask came from
    (`arena`, `harness`, `latest`); `warning` when a version was pinned and another one ran.
    """
    try:
        detected = (agent_dir / FILE).read_text().strip() or None
    except OSError:
        detected = None
    line = {"requested": requested, "detected": detected, "origin": origin}
    # `v1.1.0` asked for and `1.1.0` printed is one version; so is `2.1.326` and `2.1.326 (Claude Code)`. Whole words
    # only: `2.1.3` asked for and `2.1.326` printed are two.
    words = {w.removeprefix("v") for w in re.findall(r"[\w.+-]+", detected or "")}
    if requested and requested != LATEST and detected and requested.removeprefix("v") not in words:
        line["warning"] = f"{requested} was asked for, {detected} ran"
    return {k: v for k, v in line.items() if v is not None}
