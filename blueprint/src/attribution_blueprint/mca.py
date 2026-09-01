# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCA-style configuration, modelled on Open MPI's Modular Component Architecture.

Three aligned surfaces feed one canonical settings tree:

  CLI    --connector eca slurm ids "1,5,7"     space-separated path; last token = value
  Env    NVAB_ECA_SLURM_IDS="1,5,7"            uppercase, underscore-separated
  TOML   [eca.slurm] ids = "1,5,7"             nested sections (dotted-key sugar accepted)

Frameworks: eca (alias execution_context), log (alias logs) → applog/syslog,
output, attribution, workload. Selection sets `<framework>.component`; e.g.
`--connector eca slurm` → eca.component = "slurm".

Precedence: CLI > env > TOML > built-in defaults.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from loguru import logger

from attribution_blueprint import mca_registry as _registry
from attribution_blueprint.connectors.attribution import AttributionConnector
from attribution_blueprint.connectors.logs import LogConnector
from attribution_blueprint.connectors.metrics import MetricsConnector
from attribution_blueprint.connectors.outputs import OutputConnector
from attribution_blueprint.exceptions import MCAError
from attribution_blueprint.mca_registry import _PARAM
from attribution_blueprint.types import Source

# --- Schema ------------------------------------------------------------------
ENV_PREFIX = "NVAB_"

# --- Built-in framework + component registrations ----------------------------
# Later tasks move these into the connector packages themselves; for now they
# live inline so SCHEMA can be built from the registry without circular imports.
# Builders are referenced by name and resolved at call time (lambda closure).

if "attribution" not in _registry._FRAMEWORKS:
    _registry.register_framework(
        name="attribution",
        direct_params={"service_url": _PARAM, "timeout_seconds": _PARAM},
    )
if "workload" not in _registry._FRAMEWORKS:
    _registry.register_framework(name="workload", direct_params={"gpus_per_node": _PARAM})

# Auto-discover every connector package under attribution_blueprint.connectors.
# Each package registers its framework and/or component via side-effect imports
# of its __init__.py. Dropping a new package here is sufficient to add a
# component, no edits to this file required.
_registry.discover("attribution_blueprint.connectors")

SCHEMA: dict[str, Any] = _registry.build_schema()


# --- Schema walking ----------------------------------------------------------


def _children(node: dict) -> dict:
    """Return a node's regular (non-underscore-prefixed) children."""
    return {k: v for k, v in node.items() if not k.startswith("_")}


def _resolve_alias(node: dict, name: str) -> str | None:
    """Return canonical child name for `name` under `node`, or None if no match."""
    name = name.lower()
    for canonical, child in _children(node).items():
        if name == canonical:
            return canonical
        if isinstance(child, dict) and name in child.get("_aliases", ()):
            return canonical
    return None


def _selector_match(node: dict, name: str) -> str | None:
    sel = node.get("_selector")
    if not sel:
        return None
    name = name.lower()
    return name if name in sel else None


def is_help_request(tokens: list[str]) -> bool:
    """True iff `tokens` is an MCA path ending in `help` at a section position.

    `--connector eca slurm help` is a help request, `slurm` is a section.
    `--connector eca slurm ids help` is NOT, `ids` is a _PARAM and `help` is its value.
    Malformed paths return False so the normal resolver raises the user-facing error.
    """
    if not tokens or tokens[-1] != "help":
        return False
    node: dict = SCHEMA
    for tok in tokens[:-1]:
        sel = _selector_match(node, tok)
        if sel is not None:
            node = node["_selector"][sel]
            continue
        canonical = _resolve_alias(node, tok)
        if canonical is None and "_instances" in node:
            # User-chosen instance name. Reserved-name violations fall through
            # to the resolver so it surfaces the real error.
            if not tok or tok == "component" or tok.startswith("_"):
                return False
            node = {"_selector": node["_instances"]}
            continue
        if canonical is None:
            return False
        child = node[canonical]
        if child is _PARAM or (isinstance(child, dict) and child.get("_open")):
            return False
        node = child
    return True


def format_mca_help(tokens: list[str]) -> str:
    """Render help text for the MCA path described by `tokens` (without the
    trailing `help`). Empty list -> root help."""
    node: dict = SCHEMA
    path: list[str] = []
    for tok in tokens:
        sel = _selector_match(node, tok)
        if sel is not None:
            path.append(sel)
            node = node["_selector"][sel]
            continue
        canonical = _resolve_alias(node, tok)
        if canonical is None and "_instances" in node:
            if not tok or tok == "component" or tok.startswith("_"):
                raise MCAError(f"--connector {'.'.join(path)} instance name {tok!r} is reserved")
            path.append(tok)
            node = {"_selector": node["_instances"]}
            continue
        if canonical is None:
            raise MCAError(f"unknown --connector key {tok!r} at {'.'.join(path) or '<root>'}")
        child = node[canonical]
        if child is _PARAM:
            raise MCAError(f"--connector {'.'.join(path + [canonical])} is a value parameter, not a section")
        if isinstance(child, dict) and child.get("_open"):
            raise MCAError(f"--connector {'.'.join(path + [canonical])} is an open key/value namespace")
        path.append(canonical)
        node = child
    return _render_help(path, node)


def _render_help(path: list[str], node: dict) -> str:
    location = ".".join(path) if path else "<root>"
    prefix = "--connector " + " ".join(path) if path else "--connector"
    lines = [f"MCA help: {location}", ""]

    selector = node.get("_selector")
    if selector:
        lines.append(f"Components ({prefix} <component> ...):")
        for name in selector:
            lines.append(f"  - {name}")
        lines.append("")

    instances = node.get("_instances")
    if instances:
        lines.append(f"Instances ({prefix} <instance_name> <component> ...):")
        lines.append("  instance names are user-chosen (reserved: 'component', names starting with '_')")
        lines.append("  available components:")
        for name in instances:
            lines.append(f"    - {name}")
        lines.append("")

    params: list[str] = []
    sections: list[str] = []
    opens: list[str] = []
    for name, child in _children(node).items():
        aliases = ""
        if isinstance(child, dict) and child.get("_aliases"):
            aliases = f"  (aliases: {', '.join(child['_aliases'])})"
        if child is _PARAM:
            params.append(f"  - {name}{aliases}")
        elif isinstance(child, dict):
            if child.get("_open"):
                opens.append(f"  - {name} <key> <value>{aliases}")
            else:
                sections.append(f"  - {name}{aliases}")

    if params:
        lines.append(f"Keys ({prefix} <key> <value>):")
        lines.extend(params)
        lines.append("")
    if opens:
        lines.append(f"Open namespaces ({prefix} <name> <key> <value>):")
        lines.extend(opens)
        lines.append("")
    if sections:
        lines.append(f"Sub-sections ({prefix} <name> ...):")
        lines.extend(sections)
        lines.append("")

    if not (selector or params or opens or sections):
        lines.append("(no further keys)")

    return "\n".join(lines).rstrip() + "\n"


def resolve_cli_tokens(tokens: list[str]) -> tuple[tuple[str, ...], str]:
    """Resolve `--connector` tokens to a (canonical_path, value) pair.

    Selection (`--connector eca slurm`) returns path=("eca","component"), value="slurm".
    Param      (`--connector eca slurm ids "1,5"`) returns path=("eca","slurm","ids"), value="1,5".
    Instance   (`--connector output primary sqlite path /x.db`) returns
               path=("output","primary","sqlite","path"), value="/x.db".
    """
    if not tokens:
        raise MCAError("--connector requires at least one token")
    node: dict = SCHEMA
    path: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        sel = _selector_match(node, tok)
        if sel is not None:
            if i == len(tokens) - 1:
                return tuple(path + ["component"]), sel
            path.append(sel)
            node = node["_selector"][sel]
            i += 1
            continue
        canonical = _resolve_alias(node, tok)
        if canonical is None and "_instances" in node:
            # tok is a user-chosen instance name. Reject reserved names and
            # descend into a synthesized selector node so further tokens
            # reuse the existing selector resolution path.
            if not tok or tok == "component" or tok.startswith("_"):
                raise MCAError(f"--connector {'.'.join(path) or '<root>'} instance name {tok!r} is reserved")
            if i == len(tokens) - 1:
                raise MCAError(f"--connector {'.'.join(path + [tok])} requires a component selection")
            path.append(tok)
            node = {"_selector": node["_instances"]}
            i += 1
            continue
        if canonical is None:
            raise MCAError(f"unknown --connector key {tok!r} at {'.'.join(path) or '<root>'}")
        child = node[canonical]
        if child is _PARAM:
            if i != len(tokens) - 2:
                raise MCAError(
                    f"--connector {'.'.join(path + [canonical])} expects exactly one value; got {tokens[i + 1 :]!r}"
                )
            return tuple(path + [canonical]), tokens[i + 1]
        if isinstance(child, dict) and child.get("_open"):
            if i >= len(tokens) - 2:
                raise MCAError(f"--connector {'.'.join(path + [canonical])} requires sub-key and value")
            return tuple(path + [canonical] + list(tokens[i + 1 : -1])), tokens[-1]
        path.append(canonical)
        node = child
        i += 1
    raise MCAError(f"--connector path {'.'.join(path)} is incomplete")


def _implicit_component_assignment(path: tuple[str, ...]) -> tuple[tuple[str, ...], str] | None:
    """Return the component selector implied by a component-qualified path.

    `--connector eca kueue ids foo` should both set `eca.kueue.ids=foo` and
    select `eca.component=kueue`; otherwise the parameter lands under a
    non-selected component and is ignored. The same rule applies to sourced
    frameworks (`log.applog.loki.endpoint`) and instance frameworks
    (`output.primary.sqlite.path`).
    """
    node: dict = SCHEMA
    prefix: list[str] = []
    i = 0
    while i < len(path):
        if "_instances" in node:
            if i + 1 >= len(path):
                return None
            instance_name = path[i]
            component = path[i + 1]
            if component not in node["_instances"]:
                return None
            return tuple(prefix + [instance_name, "component"]), component

        if "_selector" in node:
            component = path[i]
            if component not in node["_selector"]:
                return None
            return tuple(prefix + ["component"]), component

        key = path[i]
        child = node.get(key)
        if not isinstance(child, dict):
            return None
        prefix.append(key)
        node = child
        i += 1
    return None


# --- Env-var enumeration -----------------------------------------------------


def _walk_env(node: dict, path: tuple[str, ...], spelling: tuple[str, ...], results: list) -> None:
    """Recursively enumerate env-var keys for a schema node into `results`.

    Each result is `(env_key, canonical_path, kind)` where
    kind ∈ {"select", "param", "open", "instances"}. Used by the global
    enumerator and by `parse_env`'s dynamic per-instance resolution.
    """
    if "_selector" in node:
        results.append(("_".join(spelling).upper(), path + ("component",), "select"))
        for comp_name, comp_node in node["_selector"].items():
            _walk_env(comp_node, path + (comp_name,), spelling + (comp_name,), results)
    if "_instances" in node:
        # Instance names are user-chosen so we can't enumerate them statically.
        # Emit a sentinel so `_build_env_indexes` registers a dynamic prefix.
        results.append(("_".join(spelling).upper(), path, "instances"))
    for name, child in _children(node).items():
        if name == "_selector":
            continue
        names = (name,) + (child.get("_aliases", ()) if isinstance(child, dict) else ())
        for alt in names:
            new_spelling = spelling + (alt,)
            new_path = path + (name,)
            if child is _PARAM:
                results.append(("_".join(new_spelling).upper(), new_path, "param"))
            elif isinstance(child, dict):
                if child.get("_open"):
                    results.append(("_".join(new_spelling).upper(), new_path, "open"))
                else:
                    _walk_env(child, new_path, new_spelling, results)


def _enumerate_env_keys() -> list[tuple[str, tuple[str, ...], str]]:
    """Yield (env_key, canonical_path, kind) for every settable point including
    alias-spelled variants. kind ∈ {"select", "param", "open", "instances"}."""
    results: list[tuple[str, tuple[str, ...], str]] = []
    _walk_env(SCHEMA, (), (), results)
    return results


_ENV_LOOKUP: dict[str, tuple[tuple[str, ...], str]] = {}
_ENV_OPEN_PREFIXES: list[tuple[str, tuple[str, ...]]] = []
# Each entry: (env_prefix_with_trailing_underscore, base_path, instances_dict).
# Used to resolve env vars whose tail's first segment is a user-chosen
# instance name; the rest is then resolved against a synthesized selector.
_ENV_INSTANCE_PREFIXES: list[tuple[str, tuple[str, ...], dict]] = []


def _build_env_indexes() -> None:
    _ENV_LOOKUP.clear()
    _ENV_OPEN_PREFIXES.clear()
    _ENV_INSTANCE_PREFIXES.clear()
    # The "instances" entry stores the env_key as its sentinel and is paired
    # with the instances_dict by walking SCHEMA again here (so callers don't
    # need to thread the dict through the tuple-shaped results).
    for key, path, kind in _enumerate_env_keys():
        if kind == "open":
            _ENV_OPEN_PREFIXES.append((key + "_", path))
        elif kind == "instances":
            # Resolve the instances_dict by walking SCHEMA along `path`.
            cur: Any = SCHEMA
            for step in path:
                cur = cur[step]
            _ENV_INSTANCE_PREFIXES.append((key + "_", path, cur["_instances"]))
        else:
            _ENV_LOOKUP[key] = (path, kind)


_build_env_indexes()


def parse_env(environ: Mapping[str, str], prefix: str = ENV_PREFIX) -> dict:
    """Build a nested settings dict from environment variables."""
    out: dict = {}
    for raw_name, value in environ.items():
        if not raw_name.startswith(prefix):
            continue
        suffix = raw_name[len(prefix) :]
        match = _ENV_LOOKUP.get(suffix)
        if match is not None:
            path, _kind = match
            implied = _implicit_component_assignment(path)
            if implied is not None:
                _set_nested(out, implied[0], implied[1])
            _set_nested(out, path, value)
            continue
        matched = False
        for open_prefix, base_path in _ENV_OPEN_PREFIXES:
            if suffix.startswith(open_prefix):
                # Open namespaces (e.g. query_kwargs) hold flat string keys;
                # underscores inside the tail belong to the key, not to a path.
                tail = suffix[len(open_prefix) :].lower()
                full_path = base_path + (tail,)
                implied = _implicit_component_assignment(full_path)
                if implied is not None:
                    _set_nested(out, implied[0], implied[1])
                _set_nested(out, full_path, value)
                matched = True
                break
        if matched:
            continue
        for ins_prefix, base_path, instances_dict in _ENV_INSTANCE_PREFIXES:
            if not suffix.startswith(ins_prefix):
                continue
            tail = suffix[len(ins_prefix) :]
            instance_name, _, rest = tail.partition("_")
            instance_name = instance_name.lower()
            if not instance_name or instance_name == "component" or instance_name.startswith("_"):
                raise MCAError(f"env var {raw_name}: invalid instance name {instance_name!r}")
            synthesized = {"_selector": instances_dict}
            local: list[tuple[str, tuple[str, ...], str]] = []
            _walk_env(synthesized, (), (), local)
            sub_lookup = {k: (p, kind) for k, p, kind in local if kind != "instances"}
            sub_match = sub_lookup.get(rest)
            if sub_match is None:
                # Unknown instance-scoped key. Skip rather than abort — startup
                # shouldn't crash on a wrapper-set NVAB_ var the schema doesn't
                # know about.
                logger.warning(f"ignoring unknown env var {raw_name}: no schema match under {base_path}")
                matched = True
                break
            sub_path, _kind = sub_match
            full_path = base_path + (instance_name,) + sub_path
            implied = _implicit_component_assignment(full_path)
            if implied is not None:
                _set_nested(out, implied[0], implied[1])
            _set_nested(out, full_path, value)
            matched = True
            break
        if not matched:
            # The NVAB_ prefix is plausible for collision with CI tooling /
            # container runtimes / wrapper scripts. Warn-and-skip is safer
            # than raising MCAError, which crashes startup on any unrelated
            # NVAB_-prefixed environment variable.
            logger.warning(f"ignoring unknown env var {raw_name}: no schema match")
    return out


# --- TOML normalization ------------------------------------------------------


def _canonicalize_toml(raw: dict, node: dict = SCHEMA) -> dict:
    """Rewrite alias keys to canonical names and validate against the schema."""
    out: dict = {}
    for key, value in raw.items():
        if "_selector" in node:
            # In selector context, allow either "component" or a component name.
            if key == "component":
                if not isinstance(value, str) or value not in node["_selector"]:
                    raise MCAError(f"invalid component selection: {value!r}")
                out["component"] = value
                continue
            if key in node["_selector"]:
                sub = value if isinstance(value, dict) else {}
                out[key] = _canonicalize_toml(sub, node["_selector"][key])
                continue
            raise MCAError(f"unknown selector child {key!r}")
        if "_instances" in node:
            # Every key at an instances level is a user-chosen instance name.
            if key == "component" or key.startswith("_"):
                raise MCAError(f"instance name {key!r} is reserved")
            if not isinstance(value, dict):
                raise MCAError(f"{key} expects a [section]")
            synthesized = {"_selector": node["_instances"]}
            out[key] = _canonicalize_toml(value, synthesized)
            continue
        canonical = _resolve_alias(node, key)
        if canonical is None:
            raise MCAError(f"unknown TOML key {key!r}")
        child = node[canonical]
        if child is _PARAM:
            out[canonical] = value
        elif isinstance(child, dict) and child.get("_open"):
            if not isinstance(value, dict):
                raise MCAError(f"{canonical} expects a table of key/value pairs")
            out[canonical] = dict(value)
        elif isinstance(child, dict):
            if not isinstance(value, dict):
                raise MCAError(f"{canonical} expects a [section]")
            out[canonical] = _canonicalize_toml(value, child)
        else:
            raise MCAError(f"unexpected schema entry under {canonical!r}")
    return out


def load_toml(path: str | Path) -> dict:
    path_obj = Path(path)
    try:
        with path_obj.open("rb") as config_file:
            raw = tomllib.load(config_file)
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise MCAError(f"invalid TOML config {path_obj}: {exc}") from exc
    return _canonicalize_toml(raw)


# --- Dict merge --------------------------------------------------------------


def _set_nested(d: dict, path: tuple[str, ...], value: Any) -> None:
    cur = d
    for k in path[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[path[-1]] = value


def deep_merge(*sources: dict, _path: tuple[str, ...] = ()) -> dict:
    """Right-most source wins (per key); dicts merge recursively.

    Emits a debug log whenever a non-dict leaf is replaced by a later source so
    users can trace where an unexpected setting came from. `_path` is private
    and only used by the recursive call to surface the full dotted key.
    """
    result: dict = {}
    for src in sources:
        if not src:
            continue
        for k, v in src.items():
            sub_path = (*_path, k)
            if isinstance(v, dict) and isinstance(result.get(k), dict):
                result[k] = deep_merge(result[k], v, _path=sub_path)
            else:
                if k in result and result[k] != v:
                    logger.debug(
                        "settings overwrite {}: {!r} -> {!r}",
                        ".".join(sub_path),
                        result[k],
                        v,
                    )
                result[k] = v
    return result


def settings_from_cli(connector_invocations: list[list[str]] | None) -> dict:
    """Convert each `--connector` invocation's tokens into the canonical dict."""
    out: dict = {}
    for tokens in connector_invocations or []:
        path, value = resolve_cli_tokens(tokens)
        implied = _implicit_component_assignment(path)
        if implied is not None:
            _set_nested(out, implied[0], implied[1])
        _set_nested(out, path, value)
    return out


# --- Build Config from merged settings --------------------------------------


@dataclass
class Config:
    attribution: AttributionConnector
    execution_context_adapters: list
    outputs: list[OutputConnector]
    log_connectors: list[LogConnector]
    metrics_connectors: list[MetricsConnector]
    gpus_per_node: int | None = None


def _coerce_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise MCAError(f"{field} must be an int")
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except ValueError as exc:
        raise MCAError(f"{field}={value!r} must be an int") from exc


def _coerce_float(value: Any, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise MCAError(f"{field}={value!r} must be a float") from exc


def _coerce_datetime(value: Any, field: str) -> datetime:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise MCAError(f"{field}={value!r} must be ISO-8601") from exc


def _coerce_str_list(value: Any, field: str) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [piece.strip() for piece in value.split(",") if piece.strip()]
    raise MCAError(f"{field}={value!r} must be a list or comma-separated string")


def _component_or_raise(framework: str, name: str):
    try:
        return _registry.get_component(framework, name)
    except KeyError:
        raise MCAError(f"unknown {framework} component: {name!r}")


def build_config(merged: dict) -> Config:
    attr = merged.get("attribution") or {}
    service_url = attr.get("service_url", "http://localhost:8080")
    timeout = _coerce_float(attr.get("timeout_seconds", 60.0), "attribution.timeout_seconds")
    attribution = AttributionConnector(service_url=service_url, timeout_seconds=timeout)

    eca_spec = merged.get("eca") or {}
    component = eca_spec.get("component", "slurm")
    eca_comp = _component_or_raise("eca", component)
    eca: Any = eca_comp.build(eca_spec.get(component) or {}, ctx={"framework": "eca"})

    outputs = _build_instance_connectors(merged, framework="output")
    if not outputs:
        stdout_comp = _component_or_raise("output", "stdout")
        outputs.append(stdout_comp.build({}, ctx={"framework": "output", "instance": "default"}))

    log_connectors = _build_sourced_connectors(merged, framework="log")
    metrics_connectors = _build_sourced_connectors(merged, framework="metrics")

    workload = merged.get("workload") or {}
    gpus_per_node = workload.get("gpus_per_node")
    if gpus_per_node is not None:
        gpus_per_node = _coerce_int(gpus_per_node, "workload.gpus_per_node")
        if gpus_per_node <= 0:
            raise MCAError(f"workload.gpus_per_node must be positive, got {gpus_per_node}")

    return Config(
        attribution=attribution,
        execution_context_adapters=[eca],
        outputs=outputs,
        log_connectors=log_connectors,
        metrics_connectors=metrics_connectors,
        gpus_per_node=gpus_per_node,
    )


def _build_instance_connectors(merged: dict, *, framework: str) -> list[Any]:
    """Build per-instance components for an instances framework (`output`).

    Each user-chosen instance under `[framework.<name>]` selects one component
    via the per-instance `component` key. Returns the list of built components
    in TOML/CLI key order. Returns [] when the framework section is absent.
    """
    spec = merged.get(framework) or {}
    out: list[Any] = []
    for instance_name, inst_spec in spec.items():
        if instance_name == "component" or instance_name.startswith("_"):
            raise MCAError(f"{framework}.{instance_name!r} is a reserved instance name")
        if not isinstance(inst_spec, dict):
            raise MCAError(f"{framework}.{instance_name} must be a [section]")
        comp_name = inst_spec.get("component")
        if comp_name is None:
            raise MCAError(f"{framework}.{instance_name} requires a `component` key")
        comp_spec = _component_or_raise(framework, comp_name)
        params = inst_spec.get(comp_name) or {}
        out.append(comp_spec.build(params, ctx={"framework": framework, "instance": instance_name}))
    return out


def _build_sourced_connectors(merged: dict, *, framework: str) -> list[Any]:
    """Build per-source connectors for a sourced framework (`log`, `metrics`).

    Returns [] when the framework isn't registered or has no configured sources.
    """
    try:
        fw = _registry.get_framework(framework)
    except KeyError:
        return []
    spec = merged.get(framework) or {}
    out: list[Any] = []
    for source_name in fw.sources:
        src_spec = spec.get(source_name)
        if not src_spec:
            continue
        comp_name = src_spec.get("component")
        if comp_name is None:
            continue
        comp_spec = _component_or_raise(framework, comp_name)
        source = Source[source_name.upper()]
        out.append(
            comp_spec.build(
                src_spec.get(comp_name) or {},
                ctx={"framework": framework, "source": source, "source_name": source_name},
            )
        )
    return out


def load_settings(
    toml_path: str | Path | None = None,
    cli_invocations: list[list[str]] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict:
    """Compose the canonical settings dict from TOML + env + CLI (in increasing precedence)."""
    toml_dict = load_toml(toml_path) if toml_path else {}
    env_dict = parse_env(environ if environ is not None else os.environ)
    cli_dict = settings_from_cli(cli_invocations)
    return deep_merge(toml_dict, env_dict, cli_dict)


__all__ = [
    "Config",
    "SCHEMA",
    "ENV_PREFIX",
    "build_config",
    "deep_merge",
    "format_mca_help",
    "is_help_request",
    "load_settings",
    "load_toml",
    "parse_env",
    "resolve_cli_tokens",
    "settings_from_cli",
]


# Optional log warning hook left here for future use by callers that want to
# notify on unused TOML sections; kept silent now.
_ = logger
