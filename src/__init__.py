from pathlib import Path
import yaml

# Load default prompts from YAML file in the same directory
_prompts_file = Path(__file__).with_name("prompts.default.yaml")
with open(_prompts_file, 'r', encoding='utf-8') as f:
    default_prompts = yaml.safe_load(f)

__all__ = ['default_prompts']
