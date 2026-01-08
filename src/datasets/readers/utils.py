import re


def to_snake_case(name: str) -> str:
    """Convert a string to snake_case."""
    name = name.strip()
    name = re.sub(r"\W+", "_", name)
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.lower()
