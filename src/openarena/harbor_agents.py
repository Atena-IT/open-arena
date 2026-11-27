# License Apache 2.0: (c) 2026 Atena-Reply

"""Harbor 0.24's Goose, SWE-agent and Cline, fixed as Harbor 0.25 fixed them, for the arena's models and tasks.

Each fails every arena trial in 0.24, before any model call: Goose refuses openrouter, SWE-agent hands SWE-agent a
literal `$(pwd)` as the repository path, Cline wants `provider:model-id` where the arena passes litellm's
`provider/model`. `run_trial` runs these classes instead of Harbor's (`OWN_AGENTS`). Remove them with the bump to
Harbor 0.25 or later, which carries the three fixes (harbor-framework/harbor#3551, #3715, #3474):
`test_harbor_agents_are_for_harbor_0_24` fails on any other Harbor, so the bump has to decide.
"""

import functools
import json
import math
import re
import uuid
from pathlib import Path

from harbor.agents.installed.base import with_prompt_template
from harbor.agents.installed.cline.cline import ClineCli as HarborCline
from harbor.agents.installed.goose import Goose as HarborGoose
from harbor.agents.installed.swe_agent import SweAgent as HarborSweAgent

# The providers Harbor hands to Goose; 0.24 leaves out openrouter, which Goose itself serves (OPENROUTER_API_KEY).
GOOSE_PROVIDERS = frozenset({"anthropic", "databricks", "google", "openai", "openrouter", "tetrate"})
# The providers each agent can call, as Harbor names them: Cline's are those whose key Harbor 0.24 hands it.
PROVIDERS = {"goose": GOOSE_PROVIDERS, "cline-cli": frozenset(HarborCline.PROVIDER_API_KEY_ENVS)}

# Harbor 0.24's repository argument: single quotes keep `$(pwd)` literal, and SWE-agent fails on `/app/$(pwd)`.
SWE_REPO_0_24 = (
    "$(if [ -d /testbed ]; then echo '--env.repo.type=preexisting --env.repo.repo_name=/testbed'; "
    "else echo '--env.repo.path=$(pwd)'; fi)"
)
# Harbor 0.25's (harbor-framework/harbor#3715): the working directory, as it is, like /testbed.
SWE_REPO = '--env.repo.type=preexisting "--env.repo.repo_name=$(if [ -d /testbed ]; then echo /testbed; else pwd; fi)"'
# SWE-agent resets the repository and builds its patch with git, and an arena task's folder is rarely a repository
# (harbor-framework/harbor#3713, left open upstream): it becomes one, its files committed, so the reset keeps them. The reset's
# `git restore .` fails on a commit without a file, so an empty folder gets a `.gitkeep`. Only a repository rooted here
# counts: SWE-agent would reset a parent one whole. The verifier then finds a `.git` (and maybe a `.gitkeep`) in the
# folder: harmless, since a task whose tests read git state ships a repository, for which nothing is made.
SWE_GIT_INIT = (
    '[ -d /testbed ] || [ "$(git rev-parse --show-toplevel 2>/dev/null)" = "$(pwd -P)" ] || { git init -q && git add -A && '
    '{ [ -n "$(git ls-files)" ] || { touch .gitkeep && git add .gitkeep; }; } && '
    "git -c user.name=arena -c user.email=arena@localhost commit -qm task; } || "
    'echo "arena: could not make $(pwd) a git repository with its files committed, which SWE-agent needs" >&2\n'
)


class Goose(HarborGoose):
    # Replaces harbor/agents/installed/goose.py `Goose.run` (Harbor 0.24.0, lines 731-799).
    @with_prompt_template
    async def run(self, instruction, environment, context) -> None:
        # Harbor 0.24's run, with openrouter among the providers (harbor-framework/harbor#3551), and a heredoc
        # delimiter the recipe cannot hold (Harbor's is `EOF`).
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("Model name must be in the format provider/model_name")
        provider, model = self.model_name.split("/", 1)
        access = self.model_connection
        provider = access.provider or provider
        if provider not in GOOSE_PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}")
        if not access.api_key:
            raise ValueError(f"No API key found for provider: {provider}")
        if provider == "databricks" and not access.configured_base_url:
            raise ValueError("DATABRICKS_HOST environment variable not set")
        env = {
            **access.env,
            "GOOSE_MODEL": model,
            "GOOSE_PROVIDER": provider,
            "XDG_DATA_HOME": "/logs/agent/goose/xdg-data",
            "XDG_STATE_HOME": "/logs/agent/goose/xdg-state",
        }
        env.update(self.resolve_env_vars())
        if skills_command := self._build_register_skills_command():
            await self.exec_as_agent(environment, command=skills_command, env=env, timeout_sec=10)
        recipe_yaml = self._create_recipe_yaml(instruction)
        end = f"HARBOR_RECIPE_{uuid.uuid4().hex}"
        await self.exec_as_agent(environment, command=f"cat > ~/harbor-recipe.yaml << '{end}'\n{recipe_yaml}{end}", env=env, timeout_sec=10)
        resume_flag = "--resume " if self._resume else ""
        cli_flags = self.build_cli_flags()
        extra_flags = (cli_flags + " ") if cli_flags else ""
        await self.exec_as_agent(
            environment,
            command=(
                'export PATH="$HOME/.local/bin:$PATH" && goose run --recipe ~/harbor-recipe.yaml '
                f"{resume_flag}--output-format stream-json {extra_flags}2>&1 | stdbuf -oL tee /logs/agent/goose.txt"
            ),
            env=env,
        )


def _count(value) -> float | None:
    """A usage number SWE-agent wrote, or None for anything that is not one (a string, a bool, a negative, an infinity)."""
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    return value if ok else None


def _model_stats(agent: "SweAgent") -> dict | None:
    """`info.model_stats` of the trial's SWE-agent trajectory, or None: none written, unreadable, cut short, another shape."""
    try:
        path = agent._find_trajectory_file() or Path(agent.logs_dir) / "swe-agent.trajectory.json"
        info = json.loads(Path(path).read_text()).get("info")
    except (OSError, ValueError, AttributeError):
        return None
    stats = info.get("model_stats") if isinstance(info, dict) else None
    return stats if isinstance(stats, dict) else None


@functools.lru_cache(maxsize=64)
def _unit_price(model: str) -> tuple[float, float] | None:
    """The arena's whole price per token for `model` (`lookups.prices.model_price`), once per process: its first read may wait
    on OpenRouter, and every SWE-agent trial of a run asks the same model."""
    try:
        from openarena.lookups.prices import model_price

        price = model_price(model)
    except Exception:  # noqa: BLE001 - a price unread costs the cell its cost, never the trial
        return None
    return (price.input, price.output) if price is not None and price.complete else None


def _priced(model: str | None, tokens_in: int | None, tokens_out: int | None) -> float | None:
    """Tokens at the arena's price for `model`, or None when no source has a whole price or a side's tokens are unknown."""
    if not model or tokens_in is None or tokens_out is None or not (tokens_in or tokens_out) or not (price := _unit_price(model)):
        return None
    return tokens_in * price[0] + tokens_out * price[1]


# A version SWE-agent's package says (`1.1.0`): anything else its command printed (a traceback) is no version.
VERSION = re.compile(r"^\d+(\.\d+)+\S*$")


class SweAgent(HarborSweAgent):
    # Harbor 0.24's finder asserts one trajectory at most, which would raise into the trial: the newest one is read.
    def _find_trajectory_file(self) -> Path | None:
        try:
            found = sorted(Path(self.logs_dir, "swe-agent-output").glob("**/*.traj"), key=lambda p: p.stat().st_mtime)
        except OSError:
            return None
        return found[-1] if found else None

    def parse_version(self, stdout: str) -> str:
        lines = stdout.strip().splitlines()
        return lines[-1].strip() if lines and VERSION.match(lines[-1].strip()) else ""

    # harbor/agents/installed/swe_agent.py asks `pip show swe-agent`: the venv uv made has no pip, and the package is `sweagent`.
    # The venv is Harbor's install's; a SWE-agent the image brings elsewhere is read with the PATH's python.
    def get_version_command(self) -> str | None:
        return (
            'p=/opt/sweagent-venv/bin/python; [ -x "$p" ] || p=python3; '
            "\"$p\" -c \"import importlib.metadata as m; print(m.version('sweagent'))\""
        )

    # Harbor 0.24 reads `info.total_cost`, `input_tokens` and `output_tokens`; SWE-agent 1.x writes its usage under
    # `info.model_stats` (`instance_cost`, `tokens_sent`, `tokens_received`). Harbor's numbers stand when it found any. The
    # cost is SWE-agent's own when above 0, else its tokens at the arena's price: litellm, which SWE-agent asks, has no price
    # for an `openrouter/...` id and SWE-agent then writes 0. Nothing reported stays None, never 0.
    def populate_context_post_run(self, context) -> None:
        super().populate_context_post_run(context)
        stats = _model_stats(self) or {}
        sent, received, cost = (_count(stats.get(key)) for key in ("tokens_sent", "tokens_received", "instance_cost"))
        context.n_input_tokens = context.n_input_tokens or (int(sent) if sent is not None else None)
        context.n_output_tokens = context.n_output_tokens or (int(received) if received is not None else None)
        if not context.cost_usd:
            context.cost_usd = cost if cost else _priced(self.model_name, context.n_input_tokens, context.n_output_tokens)

    # Replaces the repository argument of harbor/agents/installed/swe_agent.py `SweAgent.run` (Harbor 0.24.0, lines 567-568).
    async def exec_as_agent(self, environment, command, env=None, cwd=None, timeout_sec=None):
        if "sweagent run" in command:
            if SWE_REPO_0_24 not in command:  # a Harbor that words it otherwise: the fix would do nothing, silently
                raise RuntimeError(f"SWE-agent's command no longer holds Harbor 0.24's repository argument {SWE_REPO_0_24!r}: "
                                   "update openarena.harbor_agents.SweAgent")
            command = SWE_GIT_INIT + command.replace(SWE_REPO_0_24, SWE_REPO)
        return await super().exec_as_agent(environment, command, env=env, cwd=cwd, timeout_sec=timeout_sec)


class ClineCli(HarborCline):
    # Feeds harbor/agents/installed/cline/cline.py `create_run_agent_commands` (Harbor 0.24.0, line 1060) the id it splits.
    def create_run_agent_commands(self, instruction):
        # Harbor 0.24 splits the id at its first `:`; the arena's `openrouter/qwen/qwen3:free` is `openrouter:qwen/qwen3:free`
        # to it (harbor-framework/harbor#3474). An id already in that form (no `/` before its `:`) goes as it is.
        name = self.model_name
        if name and "/" in name.partition(":")[0]:
            self.model_name = name.replace("/", ":", 1)
        try:
            return super().create_run_agent_commands(instruction)
        finally:
            self.model_name = name
