# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from attribution_blueprint import mca_registry
from attribution_blueprint.exceptions import MCAError
from attribution_blueprint.mca import (
    ENV_PREFIX,
    build_config,
    deep_merge,
    format_mca_help,
    is_help_request,
    load_toml,
    parse_env,
    settings_from_cli,
)
from attribution_blueprint.orchestrator import AttributionOrchestrator

# The four keys that mark an ECA component as one of the "runtime-selectable"
# kind (Slurm, Kueue, ...): a component is runtime-selectable if its schema
# declares all four. Static-style adapters declare none of them. The CLI uses
# this introspection instead of hard-coding component names so adding a third
# scheduler is a single register_component call, not a cli.py edit.
_ECA_RUNTIME_KEYS = ("ids", "start_time", "end_time", "nodelist")


def _runtime_eca_components() -> list[str]:
    """Names of all registered eca components with the ids/time-range surface."""
    return [comp.name for comp in mca_registry.components("eca") if all(k in comp.schema for k in _ECA_RUNTIME_KEYS)]


def _all_eca_component_names() -> list[str]:
    return [comp.name for comp in mca_registry.components("eca")]


DEFAULT_CONFIG_PATH = Path("./attribution-blueprint.toml")


def _configure_logging() -> None:
    """Reset the CLI stderr sink, honoring the current ATTRIBUTION_LOG_LEVEL."""
    logger.remove()
    logger.add(sys.stderr, level=os.environ.get("ATTRIBUTION_LOG_LEVEL", "INFO"))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nvidia-attribution-blueprint",
        description=(
            "Run attribution against any registered execution-context source. "
            "Configuration follows the MCA convention shared across CLI, env vars, and TOML:\n"
            f"  CLI:  --connector <framework> [<component>] <key> <value>\n"
            f"  Env:  {ENV_PREFIX}<FRAMEWORK>_[<COMPONENT>_]<KEY>=<value>\n"
            "  TOML: [framework.component]\\n key = value\n"
            "Selection: --connector eca slurm    (sets eca.component=slurm)\n"
            'Param:     --connector eca slurm ids "1,5,7"   (sets eca.slurm.ids)'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--connector",
        nargs="+",
        action="append",
        metavar="TOKEN",
        help=(
            "MCA-style assignment. Repeatable. The final token is the value; "
            "earlier tokens form the path. Aliases (eca/execution_context, log/logs) "
            "are accepted. End any path with `help` to list valid keys/components "
            "at that level (e.g. `--connector eca slurm help`)."
        ),
    )
    parser.add_argument(
        "--config",
        action="append",
        help=(
            "Path to TOML config. Repeatable. Defaults to ./attribution-blueprint.toml "
            "if no --config or --configs is provided and that file is present."
        ),
    )
    parser.add_argument(
        "--configs",
        action="append",
        metavar="DIR",
        help="Directory of TOML configs to run. Repeatable. Runs all top-level *.toml files in sorted order.",
    )
    parser.add_argument(
        "--dump-observations",
        metavar="DIR",
        help="Write each observation body to DIR/<ctx-id>.<source>.json before submitting.",
    )
    return parser


def _resolve_config_paths(
    config_args: list[str] | None,
    configs_args: list[str] | None,
    parser: argparse.ArgumentParser,
) -> list[Path | None]:
    """Return every config path requested by the CLI.

    Explicit `--config` files and top-level `--configs` directory entries are
    validated before any run starts. With no config flags, the default config is
    used when present; otherwise `[None]` signals running without a TOML file.
    """
    paths = []
    for arg in config_args or []:
        config_path = Path(arg)
        if not config_path.exists():
            parser.error(f"--config file does not exist: {config_path}")
        if not config_path.is_file():
            parser.error(f"--config path is not a file: {config_path}")
        paths.append(config_path)
    for dir_arg in configs_args or []:
        config_dir = Path(dir_arg)
        if not config_dir.exists():
            parser.error(f"--configs directory does not exist: {config_dir}")
        if not config_dir.is_dir():
            parser.error(f"--configs path is not a directory: {config_dir}")
        toml_paths = sorted(path for path in config_dir.iterdir() if path.is_file() and path.suffix == ".toml")
        if not toml_paths:
            parser.error(f"--configs directory contains no top-level .toml files: {config_dir}")
        paths.extend(toml_paths)
    if paths:
        return paths
    return [DEFAULT_CONFIG_PATH] if DEFAULT_CONFIG_PATH.exists() else [None]


def _validate_eca_runtime(merged: dict, cli_dict: dict, parser: argparse.ArgumentParser) -> None:
    """Enforce cross-field rules on eca runtime params (ids/time-range/nodelist).

    Components are partitioned by introspection of their registered schema:
    a component with the four runtime keys (`ids`, `start_time`, `end_time`,
    `nodelist`) follows the time-range cross-field rules; anything else
    (e.g. the `static` ECA, which has its own required-field validation in
    mca.py) errors on any CLI runtime params under it.

    Rules are checked against the merged dict (so TOML defaults satisfy
    requirements), but "wrong section" checks look at the CLI dict only so a
    TOML config documenting multiple ECA blocks doesn't error."""
    eca = merged.get("eca") or {}
    component = eca.get("component", "slurm")
    cli_eca = cli_dict.get("eca") or {}

    runtime_components = set(_runtime_eca_components())

    def _cli_runtime(name: str) -> dict:
        return {k: v for k, v in (cli_eca.get(name) or {}).items() if k in _ECA_RUNTIME_KEYS}

    # Reject CLI params addressed at any non-selected component.
    for other in _all_eca_component_names():
        if other == component:
            continue
        if other in runtime_components:
            if _cli_runtime(other):
                parser.error(f"eca.{other}.* runtime params are {other}-only; selected eca component is {component!r}")
        else:
            if cli_eca.get(other):
                parser.error(f"eca.{other}.* requires eca={other}")

    if component in runtime_components:
        section = eca.get(component) or {}
        ids = section.get("ids")
        start_time = section.get("start_time")
        end_time = section.get("end_time")
        nodelist = section.get("nodelist")
        if ids is not None and start_time is not None:
            parser.error(f"eca.{component}.ids and eca.{component}.start_time are mutually exclusive")
        if start_time is not None and end_time is None:
            parser.error(f"eca.{component}.start_time requires eca.{component}.end_time")
        if end_time is not None and start_time is None:
            parser.error(f"eca.{component}.end_time requires eca.{component}.start_time")
        if nodelist is not None and (start_time is None or end_time is None):
            parser.error(f"eca.{component}.nodelist is only valid with time-range mode (start_time/end_time)")
    # No ids and no start_time is allowed: the orchestrator routes to
    # eca.discover() (which returns [] by default for runtime-selectable
    # components) and the pipeline runs to completion with zero verdicts.
    # This is the documented loopback-demo path.


def _parse_ids(value) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    return [piece.strip() for piece in str(value).split(",") if piece.strip()]


def _parse_dt(value) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value))
    # ISO inputs without an explicit offset are interpreted as UTC. Naive
    # datetimes can't flow to the service (AwareDatetime contract) and would
    # silently mis-anchor sacct queries when the BP host's local tz differs
    # from slurmctld's — making UTC the explicit, documented default.
    if dt.tzinfo is None:
        from datetime import timezone

        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _run_config(
    *,
    toml_path: Path | None,
    env_dict: dict,
    cli_dict: dict,
    dump_observations: str | None,
    parser: argparse.ArgumentParser,
    aggregate_runs: list[tuple[dict[str, int], list[dict[str, str]]]] | None = None,
) -> int:
    """Run attribution once for a resolved config path.

    `toml_path=None` is the no-config-file sentinel used by the default
    loopback/demo path when no TOML config is present.

    Args:
        toml_path: Config path to run, or None for the no-config default.
        env_dict: Settings parsed from environment variables.
        cli_dict: Settings parsed from CLI connector overrides.
        dump_observations: Optional directory for observation debug dumps.
        parser: Parser used to report configuration errors.
        aggregate_runs: Optional collector for multi-config aggregate logging.

    Returns:
        Process-style return code for this config run.
    """
    try:
        toml_dict = load_toml(toml_path) if toml_path else {}
    except MCAError as e:
        parser.error(str(e))

    merged = deep_merge(toml_dict, env_dict, cli_dict)

    _validate_eca_runtime(merged, cli_dict, parser)

    try:
        config = build_config(merged)
    except MCAError as e:
        parser.error(str(e))

    eca_spec = merged.get("eca") or {}
    component = eca_spec.get("component", "slurm")

    if component in _runtime_eca_components():
        section = eca_spec.get(component) or {}
        effective_ids = _parse_ids(section["ids"]) if section.get("ids") is not None else None
        since = _parse_dt(section["start_time"]) if section.get("start_time") is not None else None
        until = _parse_dt(section["end_time"]) if section.get("end_time") is not None else None
        # Always wrap the raw value as a single-element list. Each adapter's
        # discover_by_time_range parses its own input syntax (slurm expands
        # bracket expressions; kueue CSV-splits plain k8s node names), so the
        # CLI no longer has to know per-component nodelist semantics.
        nodelist = [section["nodelist"]] if section.get("nodelist") is not None else None
    else:
        effective_ids: list[str] | None = None
        since: datetime | None = None
        until: datetime | None = None
        nodelist: list[str] | None = None

    pipeline = AttributionOrchestrator(
        config.execution_context_adapters,
        config.log_connectors,
        config.metrics_connectors,
        config.attribution,
        config.outputs,
        dump_observations_dir=Path(dump_observations) if dump_observations else None,
        gpus_per_node=config.gpus_per_node,
    )

    try:
        pipeline.analyze(ids=effective_ids, since=since, until=until, nodelist=nodelist)
    except FileNotFoundError as e:
        missing = e.filename if e.filename is not None else str(e)
        print(f"error: file not found: {missing}", file=sys.stderr)
        return 1
    except Exception as e:
        logger.exception(f"attribution run failed: {e}")
        return 1
    # Per-context failures are contained in the orchestrator; check the
    # counters it exposes so total failure (every ctx errored, or sqlite
    # couldn't persist any verdict) surfaces as exit code 1 instead of 0.
    stats = pipeline.last_run_stats
    attempted = stats.get("attempted", 0)
    succeeded = stats.get("succeeded", 0)
    summaries = pipeline.last_run_summaries if isinstance(pipeline.last_run_summaries, list) else []
    if aggregate_runs is not None:
        aggregate_runs.append((stats, summaries))
    _write_run_summary(stats, summaries)
    if attempted > 0 and succeeded == 0:
        logger.error(f"attribution run failed: 0/{attempted} contexts succeeded")
        return 1
    return 0


def _write_aggregate_run_summary(run_results: list[tuple[dict[str, int], list[dict[str, str]]]]) -> None:
    """Log final completion totals for a multi-config run.

    Args:
        run_results: Per-config stats and summaries captured from `_run_config`.
    """

    attempted = 0
    succeeded = 0
    for stats, _summaries in run_results:
        attempted += stats.get("attempted", 0)
        succeeded += stats.get("succeeded", 0)
    _write_run_summary({"attempted": attempted, "succeeded": succeeded}, [])


def _write_run_summary(stats: dict[str, int], summaries: list[dict[str, str]]) -> None:
    """Log a concise per-run completion summary after an attribution attempt."""
    attempted = stats.get("attempted", 0)
    succeeded = stats.get("succeeded", 0)
    if attempted == 0:
        logger.info("Attribution complete: 0 execution contexts discovered.")
        return
    logger.info(f"Attribution complete: {succeeded}/{attempted} execution contexts succeeded.")
    for summary in summaries:
        attempt = summary.get("attempt_id")
        if attempt is None or attempt == "":
            attempt = "n/a"
        logger.info(
            f"- {summary.get('job_id', '?')} attempt={attempt} "
            f"terminal_state={summary.get('terminal_state', 'unknown')} "
            f"sub_domain={summary.get('sub_domain', 'unclassified')} "
            f"root_symptom={summary.get('root_symptom', 'none')} "
            f"symptoms={summary.get('symptoms', 'none')}"
        )


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)

    for tokens in args.connector or []:
        if is_help_request(tokens):
            try:
                sys.stdout.write(format_mca_help(tokens[:-1]))
            except MCAError as e:
                parser.error(str(e))
            return 0

    try:
        config_paths = _resolve_config_paths(args.config, args.configs, parser)
        env_dict = parse_env(os.environ)
        cli_dict = settings_from_cli(args.connector)
    except MCAError as e:
        parser.error(str(e))

    aggregate_runs: list[tuple[dict[str, int], list[dict[str, str]]]] | None = []
    if len(config_paths) <= 1:
        aggregate_runs = None

    for toml_path in config_paths:
        rc = _run_config(
            toml_path=toml_path,
            env_dict=env_dict,
            cli_dict=cli_dict,
            dump_observations=args.dump_observations,
            parser=parser,
            aggregate_runs=aggregate_runs,
        )
        if rc != 0:
            return rc
    if aggregate_runs is not None:
        _write_aggregate_run_summary(aggregate_runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
