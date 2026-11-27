# License Apache 2.0: (c) 2026 Atena-Reply

"""Harbor's Hermes agent, with an install today's Hermes passes.

Harbor 0.24 installs Hermes under `~/.hermes` from the installer on Hermes's `main`, runs it under
`HERMES_HOME=/tmp/hermes` and checks it with `hermes version`, a command Hermes dropped (harbor-framework/harbor#3091).
Under that second root today's `hermes chat` fails at once (`No module named 'ruamel'`), and on a 1 GiB sandbox the
installer itself exits 1: its web UI build runs out of memory, a product `hermes chat` does not use and no installer
option skips. So Hermes is installed where the trial runs it, from a pinned release, and the hermes command, not the
installer's exit code, says whether it works. `run_trial` runs this class for `hermes`. Remove it once Harbor's own
Hermes installs.
"""

import shlex

from harbor.agents.installed.hermes import Hermes as HarborHermes

HOME = "/tmp/hermes"  # the root Harbor's run exports
# Bumped on purpose: the installer and the tree it checks out, Hermes v0.23.4.
RELEASE, COMMIT = "v0.23.4", "8fc0374447313e276f6b9a2764bfc750190c966a"
INSTALL_SH = f"https://raw.githubusercontent.com/NousResearch/hermes-agent/{RELEASE}/scripts/install.sh"
LOG = "/tmp/hermes-install.log"
# The installer's last failure lines, cut: a platform keeps only the end of an error.
SAID = f"{{ grep -iE 'fail|error' {LOG} || tail -n 3 {LOG}; }} | tail -n 3 | cut -c1-200"


class Hermes(HarborHermes):
    # The installer took 10 to 11 min on one local core with 1 GiB, and more than Harbor's 1200 s on a 1 vCPU E2B sandbox.
    DEFAULT_SETUP_TIMEOUT_SEC = 1800.0

    def get_version_command(self) -> str | None:
        return f'export PATH="$HOME/.local/bin:$PATH" HERMES_HOME={HOME}; hermes --version'

    def parse_version(self, stdout: str) -> str:
        # `--version` goes on with the install directory, Python and SDK versions: the first line names the release.
        return stdout.strip().splitlines()[0] if stdout.strip() else ""

    async def install(self, environment) -> None:
        await self.ensure_system_dependencies(environment, ("curl", "git", "ripgrep", "xz"))
        # A version asked for in the trial's config is a branch or tag, as in Harbor; else the pinned release.
        tree = f"--branch {shlex.quote(self._version)}" if self._version else f"--commit {COMMIT}"
        await self.exec_as_agent(
            environment,
            command="; ".join((
                "set -uo pipefail",
                f'export HERMES_HOME={HOME} PATH="$HOME/.local/bin:$PATH"',
                f'curl -fsSL {INSTALL_SH} | bash -s -- --skip-setup {tree} 2>&1 | tee {LOG}',
                "rc=$?",
                f'[ "$rc" -eq 0 ] || {{ echo "install.sh exited $rc, the hermes command decides; it reported:" >&2; {SAID} >&2; }}',
                # The installer's example .env (TERMINAL_TIMEOUT=60, ...) is not Harbor's: the run's config.yaml sets Hermes.
                'rm -f "$HERMES_HOME/.env"',
                'mkdir -p "$HERMES_HOME/sessions" "$HERMES_HOME/skills" "$HERMES_HOME/memories"',
                "hermes sessions list >/dev/null && hermes --version && exit 0",
                f'echo "the hermes command does not run after install.sh (exit $rc), which reported:" >&2; {SAID} >&2; exit 1',
            )),
        )
