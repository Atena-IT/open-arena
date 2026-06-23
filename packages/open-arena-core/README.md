# open-arena-core

Lightweight package containing the Pydantic models (generated from `openapi.yaml`),
the `ArenaAPIClient`, and shared constants for Open Arena.

Install for thin remote-only usage:

```bash
pip install open-arena-core
```

`models.py` is generated from the root `openapi.yaml` via:

```bash
datamodel-codegen --input openapi.yaml --input-file-type openapi \
    --output packages/open-arena-core/open_arena_core/models.py \
    --output-model-type pydantic_v2.BaseModel
```
