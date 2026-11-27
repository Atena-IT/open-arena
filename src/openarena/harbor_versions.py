# License Apache 2.0: (c) 2026 Atena-Reply

"""Harbor's installed agents, each writing down the version it runs once installed, pinned or not.

Harbor 0.24 runs an agent's version command after its install only when no version was asked for: with a pin, the
trial's `agent_info.version` is the pin, whatever ran. Each class here runs that command after every install and writes
what it says to `version.txt` in the agent's logs, which the trial's line reads (`version_line`). `run_trial` runs these
classes for the built-in agents Harbor installs. Remove once Harbor checks a pinned install.
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from harbor.agents.factory import AgentFactory
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.models.agent.name import AgentName
from harbor.utils.import_path import import_class

from openarena.config.entries import HARBOR_AGENTS
from openarena.programs.harbor_program import OWN_AGENTS

FILE = "version.txt"
# How long reading the version may take once the agent is installed: a command that hangs is given up, never the trial.
VERSION_TIMEOUT = 10
# What is kept of what it printed: one line, no control characters, this long at most.
MAX_VERSION = 120
# A release (`1.6.0`, `v0.23.4`), which a printed version can be checked against; a branch or a commit cannot.
RELEASE = re.compile(r"^v?\d+(\.\d+)+")

log = logging.getLogger("uvicorn.error")


def one_line(text: str) -> str:
    """`text`'s last line, without control characters, cut to `MAX_VERSION`: what a trial's line keeps of a version."""
    lines = text.strip().splitlines()
    # Colours go whole (`\x1b[32m`), then any other control character.
    plain = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", lines[-1] if lines else "")
    return re.sub(r"[\x00-\x1f\x7f]", "", plain).strip()[:MAX_VERSION]


class Detects:
    """Mixed in before a Harbor installed agent: after its setup, what its version command says goes to `FILE`."""

    async def setup(self, environment) -> None:
        await super().setup(environment)
        command = self.get_version_command()
        if not command:
            return
        try:
            said = await asyncio.wait_for(environment.exec(command=command, timeout_sec=VERSION_TIMEOUT), VERSION_TIMEOUT * 1.5)
            # Some agents print warnings before it (OpenHands: grpc's FutureWarning): the last line is the version.
            if said.return_code == 0 and (version := one_line(self.parse_version(said.stdout or ""))):
                (Path(self.logs_dir) / FILE).write_text(version)
        except Exception as e:  # noqa: BLE001 - best effort, as Harbor's own: a trial never fails on reading a version
            log.warning("Could not read %s's version after its install: %s", type(self).__name__, e or type(e).__name__)


def _attr(agent: str) -> str:
    return re.sub(r"\W", "_", agent)


def _classes() -> dict[str, type]:
    """One class per built-in agent Harbor installs, named after it (`claude_code`), over the engine's own class where it
    has one (`OWN_AGENTS`: Hermes, Goose, SWE-agent, Cline). An agent this Harbor does not ship is left out, said in the
    log: it runs as Harbor names it, its version unread."""
    classes = {}
    for agent in HARBOR_AGENTS:
        try:
            base = import_class(OWN_AGENTS[agent], label="agent") if agent in OWN_AGENTS else AgentFactory.get_agent_class(AgentName(agent))
        except Exception as e:  # noqa: BLE001 - one agent that will not import costs its version, never every trial
            log.warning("No class for Harbor agent %r (%s): its version is not read after install.", agent, e)
            continue
        if issubclass(base, BaseInstalledAgent):
            classes[agent] = type(base.__name__, (Detects, base), {"__module__": __name__})
    return classes


CLASSES = _classes()
# Each class importable by its name; no agent's name may shadow this module's own (`re`, `log`...).
assert not set(map(_attr, CLASSES)) & set(globals()), "an agent's name shadows a name of openarena.harbor_versions"
globals().update({_attr(agent): cls for agent, cls in CLASSES.items()})


def agent_path(agent: str) -> str:
    """What a trial names for `agent`: the class here for a built-in agent Harbor installs, else `agent` as written."""
    return f"{__name__}:{_attr(agent)}" if agent in CLASSES else agent


def version_line(requested: str | None, origin: str | None, agent_dir: Path) -> dict[str, Any]:
    """A trial's `agent_version`: the version asked for, the one that ran (None when unread) and where the ask came from
    (`arena`, `harness`, `latest`); `warning` when a version was pinned and another one ran.
    """
    try:
        detected = one_line((agent_dir / FILE).read_text()) or None
    except OSError:
        detected = None
    line = {"requested": requested, "detected": detected, "origin": origin}
    # `v1.1.0` asked for and `1.1.0` printed is one version; so is `2.1.326` and `2.1.326 (Claude Code)`. Whole words
    # only: `2.1.3` asked for and `2.1.326` printed are two. A branch or a commit asked for is not what a version prints.
    words = {w.removeprefix("v") for w in re.findall(r"[\w.+-]+", detected or "")}
    if requested and RELEASE.match(requested) and detected and requested.removeprefix("v") not in words:
        line["warning"] = f"{requested} was asked for, {detected} ran"
    return {k: v for k, v in line.items() if v is not None}
