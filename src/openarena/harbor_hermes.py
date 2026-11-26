# License Apache 2.0: (c) 2026 Atena-Reply

"""Harbor's Hermes agent, with an install check today's Hermes passes.

Harbor 0.24 ends Hermes's install with `hermes version`, a command Hermes dropped (NousResearch/hermes-agent#90714),
so every Hermes trial failed there, exit 2 (harbor-framework/harbor#3091). `hermes --version` alone, Harbor's proposed
fix, still exits 1 under the data root Harbor sets (`HERMES_HOME=/tmp/hermes`): that root's dependencies are prepared by
the first command that needs them, and `hermes sessions list` is one. `run_trial` runs this class for `hermes`.
Remove it once Harbor's own Hermes installs.
"""

from typing import Any

from harbor.agents.installed.hermes import Hermes as HarborHermes

STALE_CHECK = "&& hermes version"
CHECK = "&& hermes sessions list >/dev/null && hermes --version"


class Hermes(HarborHermes):
    def get_version_command(self) -> str | None:
        return 'export PATH="$HOME/.local/bin:$PATH"; hermes --version | head -n 1'

    async def exec_as_agent(self, environment, command: str, *args: Any, **kwargs: Any) -> Any:
        # Only the install command ends so; once Harbor fixes its own, nothing matches and this passes it through.
        if command.endswith(STALE_CHECK):
            command = command.removesuffix(STALE_CHECK) + CHECK
        return await super().exec_as_agent(environment, command, *args, **kwargs)
