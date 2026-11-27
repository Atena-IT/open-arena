# License Apache 2.0: (c) 2026 Atena-Reply

"""Harbor's Hermes agent, with an install today's Hermes passes.

Harbor 0.24 installs Hermes under `~/.hermes` from the installer on Hermes's `main`, runs it under
`HERMES_HOME=/tmp/hermes` and checks it with `hermes version`, a command Hermes dropped (harbor-framework/harbor#3091).
Under that second root today's `hermes chat` fails at once (`No module named 'ruamel'`), and on a 1 GiB sandbox the
installer itself exits 1: its web UI build runs out of memory, a product `hermes chat` does not use and no installer
option skips. So Hermes is installed from a pinned release on the sandbox's disk, `/tmp/hermes` links to it, and the
hermes command, not the installer's exit code, says whether it works. `run_trial` runs this class for `hermes`. Remove it once Harbor's own
Hermes installs.
"""

import shlex

from harbor.agents.installed.hermes import Hermes as HarborHermes

from openarena.config.entries import AGENT_VERSIONS

HOME = "/tmp/hermes"  # the root Harbor's run exports: here a link to the install on the sandbox's disk
# E2B's /tmp is small: a 1 GiB sandbox ran out of space there cloning Hermes. The 3 GB install goes on the disk.
DISK = "$HOME/.hermes"
# The commit each pinned release checks out (its tag's), on Hermes's main: bumping `AGENT_VERSIONS["hermes"]` without its
# commit here fails at import, so the two cannot drift apart.
COMMITS = {"v0.23.4": "8fc0374447313e276f6b9a2764bfc750190c966a"}
RELEASE = AGENT_VERSIONS["hermes"]
COMMIT = COMMITS[RELEASE]
INSTALL_SH = f"https://raw.githubusercontent.com/NousResearch/hermes-agent/{RELEASE}/scripts/install.sh"
# The installer's log, where Harbor downloads the agent's logs even when the setup fails or times out; else the home.
LOG_DIR = "/logs/agent"
# Seconds between the lines the install prints while the installer runs into its log: E2B's command stream times out
# after minutes of silence (a staging trial lost it after about 7 minutes of a silent install).
BEAT = 20
# The installer's last failure lines, each cut to its end (`exited 137`): a platform keeps only the end of an error. Only
# its own marks (`fatal:`, `✗`, `⚠`, `exited N`): a full disk prints `error: unable to write file <name>` per file, and a
# name may hold `fail`.
SAID = (
    "{ grep -E '^[[:space:]]*(fatal:|✗|⚠)|exited [0-9]+' \"$log\" || tail -n 3 \"$log\"; } | tail -n 3 | grep -oE '.{0,200}$'"
)


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
        # A version asked for in the trial's config is a branch or tag, as in Harbor; else the pinned release. Not `_version`
        # alone: a first setup fills it with what `--version` printed, which a second setup would pass as a branch.
        tree = f"--branch {shlex.quote(self._version)}" if self._version_was_pinned else f"--commit {COMMIT}"
        await self.exec_as_agent(
            environment,
            command="; ".join((
                "set -uo pipefail",
                f'export HERMES_HOME={DISK} PATH="$HOME/.local/bin:$PATH"',
                'mkdir -p "$HERMES_HOME" || { echo "cannot create $HERMES_HOME for the Hermes install" >&2; exit 1; }',
                f'log={LOG_DIR}/hermes-install.log; {{ mkdir -p {LOG_DIR} && : >"$log"; }} 2>/dev/null || {{ log="$HOME/.hermes-install.log"; : >"$log"; }}',
                # Downloaded first, to a fresh file: a failed download is its own error, with curl's message, and a page that
                # is not a script (empty, a proxy's HTML) is refused before bash runs it.
                "installer=$(mktemp)",
                f'curl -fsSL -o "$installer" {INSTALL_SH} >>"$log" 2>&1 || {{ rc=$?; echo "could not download the Hermes installer (curl exit $rc): $(tail -n 2 "$log")" >&2; exit "$rc"; }}',
                '[ "$(head -c 2 "$installer")" = "#!" ] || { echo "the Hermes installer download is empty or not a script; it begins: $(head -n 1 "$installer" | cut -c1-120)" >&2; exit 1; }',
                # A line every `BEAT` seconds while it runs, so the command is never silent; it stops with the installer.
                # The sleep's own output goes nowhere, so a stopped beat holds no pipe open.
                "( s=; trap 'kill \"$s\" 2>/dev/null; exit 0' TERM; "
                f'while :; do sleep {BEAT} >/dev/null 2>&1 & s=$!; wait "$s"; echo "hermes install: still running, $(wc -l < "$log") lines in its log"; done ) & beat=$!',
                # Into the log only: Harbor names a failed command's error from words anywhere in its output (`rate limit`
                # makes it a rate-limit error), and the installer prints thousands of file names.
                f'bash "$installer" --skip-setup {tree} >>"$log" 2>&1',
                "rc=$?",
                'kill "$beat" 2>/dev/null; wait "$beat" 2>/dev/null',
                # The installer's example .env (TERMINAL_TIMEOUT=60, ...) is not Harbor's: the run's config.yaml sets Hermes.
                'rm -f "$HERMES_HOME/.env"',
                'mkdir -p "$HERMES_HOME/sessions" "$HERMES_HOME/skills" "$HERMES_HOME/memories"',
                f'{{ rm -rf {HOME} && ln -s "$HERMES_HOME" {HOME}; }} || {{ echo "could not link {HOME} to the install in $HERMES_HOME" >&2; exit 1; }}',
                f"export HERMES_HOME={HOME}",
                # The hermes command decides, not install.sh: on 1 GiB its web UI build runs out of memory, which `hermes chat` never uses.
                f'if hermes sessions list >/dev/null && hermes --version; then [ "$rc" -eq 0 ] || {{ echo "install.sh exited $rc, but the hermes command runs; it reported:"; {SAID}; }} >&2; exit 0; fi',
                f'echo "the hermes command does not run after install.sh (exit $rc), which reported:" >&2; {SAID} >&2; exit 1',
            )),
        )
