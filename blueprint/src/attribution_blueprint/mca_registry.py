# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry primitives for MCA frameworks and components.

Connector packages call `register_framework` / `register_component` at import
time from their `__init__.py`. `mca.py` builds SCHEMA and dispatches builders
from this registry rather than from hard-coded tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from attribution_blueprint.exceptions import RegistryError

_PARAM = object()
"""Sentinel marking a schema leaf parameter. The token after a _PARAM key is the value."""


@dataclass(frozen=True)
class FrameworkSpec:
    name: str
    aliases: tuple[str, ...] = ()
    has_selector: bool = False
    sources: tuple[str, ...] = ()
    instances: bool = False
    direct_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ComponentSpec:
    framework: str
    name: str
    schema: dict[str, Any]
    build: Callable[..., Any]


_FRAMEWORKS: dict[str, FrameworkSpec] = {}
_COMPONENTS: dict[tuple[str, str], ComponentSpec] = {}


def register_framework(
    *,
    name: str,
    aliases: tuple[str, ...] = (),
    has_selector: bool = False,
    sources: tuple[str, ...] = (),
    instances: bool = False,
    direct_params: dict[str, Any] | None = None,
) -> None:
    if name in _FRAMEWORKS:
        raise RegistryError(f"framework already registered: {name!r}")
    modes = [bool(has_selector), bool(sources), bool(instances)]
    if sum(modes) > 1:
        raise RegistryError(f"framework {name!r}: has_selector, sources, and instances are mutually exclusive")
    _FRAMEWORKS[name] = FrameworkSpec(
        name=name,
        aliases=tuple(aliases),
        has_selector=has_selector,
        sources=tuple(sources),
        instances=instances,
        direct_params=dict(direct_params or {}),
    )


def register_component(
    *,
    framework: str,
    name: str,
    schema: dict[str, Any],
    build: Callable[..., Any],
) -> None:
    if framework not in _FRAMEWORKS:
        raise RegistryError(f"unknown framework: {framework!r} (register the framework first)")
    key = (framework, name)
    if key in _COMPONENTS:
        raise RegistryError(f"component already registered: {framework}.{name}")
    _COMPONENTS[key] = ComponentSpec(framework=framework, name=name, schema=dict(schema), build=build)


def get_framework(name: str) -> FrameworkSpec:
    return _FRAMEWORKS[name]


def get_component(framework: str, name: str) -> ComponentSpec:
    return _COMPONENTS[(framework, name)]


def frameworks() -> list[FrameworkSpec]:
    return list(_FRAMEWORKS.values())


def components(framework: str) -> list[ComponentSpec]:
    return [c for (fw, _), c in _COMPONENTS.items() if fw == framework]


def _selector_for(framework_name: str) -> dict:
    return {comp.name: dict(comp.schema) for comp in components(framework_name)}


def discover(package_name: str) -> list[str]:
    """Import every submodule under `package_name` so registration side effects run.

    Returns the list of fully-qualified module names that were imported.
    Modules that fail to import propagate the exception. Registration is opt-in
    per package, but a broken package shouldn't be silently skipped.
    A module already mid-initialization in `sys.modules` is returned partially
    executed, so its `register_*` calls land after the caller has built its
    schema.
    """
    import importlib
    import pkgutil

    pkg = importlib.import_module(package_name)
    imported: list[str] = []
    for module_info in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        importlib.import_module(module_info.name)
        imported.append(module_info.name)
    return imported


def build_schema() -> dict:
    """Construct the SCHEMA dict from registered frameworks + components.

    For a flat framework (direct_params, no selector), the framework node
    contains the params directly. For a selector framework, the node has a
    `_selector` block mapping component names to their schemas. For a sourced
    framework (e.g. `log` with applog/syslog), each source key holds its own
    `_selector` block populated from the same component pool. For an instances
    framework, the node has an `_instances` block; user-chosen child keys at
    that level each behave like a selector node from the same component pool.
    """
    schema: dict = {}
    for fw in frameworks():
        node: dict = {}
        if fw.aliases:
            node["_aliases"] = fw.aliases
        for key, value in fw.direct_params.items():
            node[key] = value
        if fw.sources:
            for source in fw.sources:
                node[source] = {"_selector": _selector_for(fw.name)}
        elif fw.has_selector:
            node["_selector"] = _selector_for(fw.name)
        elif fw.instances:
            node["_instances"] = _selector_for(fw.name)
        schema[fw.name] = node
    return schema
