# License Apache 2.0: (c) 2026 Atena-Reply

"""Harbor 0.24's Goose, SWE-agent and Cline, fixed as Harbor 0.25 fixed them, for the arena's models and tasks.

Each fails every arena trial in 0.24, before any model call: Goose refuses openrouter, SWE-agent hands SWE-agent a
literal `$(pwd)` as the repository path, Cline wants `provider:model-id` where the arena passes litellm's
`provider/model`. `run_trial` runs these classes instead of Harbor's (`OWN_AGENTS`). Remove them with the bump to
Harbor 0.25 or later, which carries the three fixes (harbor-framework/harbor#3551, #3715, #3474).
"""

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
# counts: SWE-agent would reset a parent one whole.
SWE_GIT_INIT = (
    '[ -d /testbed ] || [ "$(git rev-parse --show-toplevel 2>/dev/null)" = "$(pwd -P)" ] || { git init -q && git add -A && '
    '{ [ -n "$(git ls-files)" ] || { touch .gitkeep && git add .gitkeep; }; } && '
    "git -c user.name=arena -c user.email=arena@localhost commit -qm task; } || "
    'echo "arena: could not make $(pwd) a git repository with its files committed, which SWE-agent needs" >&2\n'
)


class Goose(HarborGoose):
    @with_prompt_template
    async def run(self, instruction, environment, context) -> None:
        # Harbor 0.24's run, with openrouter among the providers (harbor-framework/harbor#3551).
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
        await self.exec_as_agent(environment, command=f"cat > ~/harbor-recipe.yaml << 'EOF'\n{recipe_yaml}EOF", env=env, timeout_sec=10)
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


class SweAgent(HarborSweAgent):
    async def exec_as_agent(self, environment, command, env=None, cwd=None, timeout_sec=None):
        if SWE_REPO_0_24 in command:
            command = SWE_GIT_INIT + command.replace(SWE_REPO_0_24, SWE_REPO)
        return await super().exec_as_agent(environment, command, env=env, cwd=cwd, timeout_sec=timeout_sec)


class ClineCli(HarborCline):
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
