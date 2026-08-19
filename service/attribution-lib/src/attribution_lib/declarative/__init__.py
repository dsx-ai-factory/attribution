"""Declarative TOML extractor-rule loader.

Turns a `.toml` rule file into ExtractorRules registered on the engine.
See docs/declarative-grammar.md.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from attribution_lib.declarative.loader import load_toml_rules as load_toml_rules

__all__ = ["load_toml_rules"]


def __getattr__(name: str) -> Any:
    """Load the public loader entry point without eagerly importing its dependencies."""
    if name == "load_toml_rules":
        from attribution_lib.declarative.loader import load_toml_rules  # noqa: PLC0415

        return load_toml_rules
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
