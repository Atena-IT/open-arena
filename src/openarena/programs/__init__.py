# License Apache 2.0: (c) 2026 Atena-Reply

"""The editable program graphs.

Four entry points the trial runner in `libraries/openarena/src/openarena/evaluate.py` calls, one
per kind of dataset:

- `build_program(...)` — Generator-based eval (Q&A, classification,
  summarization, ...). One LM call per row. See `generator_program.py`.
- `build_agent(...)` — `FunctionCallingAgent`-based eval driven by MCP
  tools loaded at build time. See `agent_program.py`.
- `build_a2a(...)` — each row sent to a remote agent over A2A; the axis
  carries `a2a_agents` keys. See `a2a_program.py`.
- `build_harbor(...)` — each row a Harbor task, run by the model in an
  environment's sandbox and scored by its verifier. See `harbor_program.py`.
- `build_embedding(...)` — retrieval over the row's chunks with the
  embedding model under test. See `embedding_program.py`.
- `build_tools(...)` — one call per row with tool schemas in hand, to test
  a model's tool calling with no server behind it. See `tools_program.py`.

The trial picks one from the block the dataset declared (`agent:`, `a2a:`, `harbor:`,
`embedding:`, `tools:`, else `generator:`); everything downstream treats them all as
opaque `Program` instances.
"""

from openarena.programs.a2a_program import build_a2a
from openarena.programs.agent_program import build_agent
from openarena.programs.embedding_program import build_embedding
from openarena.programs.generator_program import build_program
from openarena.programs.harbor_program import build_harbor
from openarena.programs.tools_program import build_tools

__all__ = ["build_a2a", "build_agent", "build_embedding", "build_harbor", "build_program", "build_tools"]
