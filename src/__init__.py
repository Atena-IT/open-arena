from importlib.resources import files
import yaml

# Load default prompts from YAML file included in the package
_prompts_text = files("src").joinpath("prompts.default.yaml").read_text(encoding="utf-8")
default_prompts = yaml.safe_load(_prompts_text)

__all__ = ['default_prompts']
