# open-arena-cli

Standalone CLI for Open Arena. Thin install for remote API operations — no heavy evaluation engine required.

## Install

**Thin (remote ops only):**
```bash
pip install open-arena-cli
```

**Full (engine + local sweep + serve):**
```bash
pip install open-arena
```
Or add the server extra to the CLI package:
```bash
pip install "open-arena-cli[server]"
```

## Usage

```bash
arena env list --server http://your-arena-server:8000
arena verifier list
arena leaderboard list
arena run submit --file run.json
arena discover metric-kinds
arena request GET /v1/metric-kinds
```

Heavy commands (`arena serve`, the default sweep, `arena run submit --local`) require
`pip install open-arena` and will raise a clear error if the engine is not installed.
