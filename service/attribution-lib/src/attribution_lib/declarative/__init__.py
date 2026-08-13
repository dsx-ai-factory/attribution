"""Declarative TOML extractor-rule loader.

Turns a `.toml` rule file into ExtractorRules registered on the engine.
See docs/declarative-grammar.md.
"""

from attribution_lib.declarative.loader import load_toml_rules

__all__ = ["load_toml_rules"]
