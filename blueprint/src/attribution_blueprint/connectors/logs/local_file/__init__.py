# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read log evidence from local files on the same host as the blueprint."""

import json
import re
from pathlib import Path
from typing import Optional

from loguru import logger

from attribution_blueprint.connectors.logs import entries_from_text, log_observation
from attribution_blueprint.connectors.logs.syslog import syslog_entries_from_text
from attribution_blueprint.exceptions import MCAError
from attribution_blueprint.types import (
    ExecutionContext,
    Observation,
    Source,
    TimeseriesEntry,
    assert_aware,
)

DEFAULT_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB

# Suffix multipliers for human-readable byte sizes. Binary (1024^n) — matches
# how dd/du/free report by default and how filesystem caps are typically set.
_SIZE_SUFFIXES: dict[str, int] = {
    "": 1,
    "b": 1,
    "k": 1024,
    "kb": 1024,
    "m": 1024**2,
    "mb": 1024**2,
    "g": 1024**3,
    "gb": 1024**3,
    "t": 1024**4,
    "tb": 1024**4,
}

_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]*)\s*$")


def parse_size(value: int | str) -> int:
    """Coerce ``2`` / ``"500MB"`` / ``"2G"`` / ``"1.5gb"`` into raw bytes."""
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        raise TypeError(f"size must be int or str, got {type(value).__name__}")
    m = _SIZE_RE.match(value)
    if not m:
        raise ValueError(f"unparseable size: {value!r}")
    number, suffix = m.group(1), m.group(2).lower()
    if suffix not in _SIZE_SUFFIXES:
        raise ValueError(f"unknown size suffix {suffix!r} in {value!r}")
    return int(float(number) * _SIZE_SUFFIXES[suffix])


class LocalFileLogConnector:
    """Collector that reads one or more log files from disk and emits one
    Observation per file.

    Each entry in ``log_path_templates`` may be a literal path, or a path with
    Python-format placeholders (``{std_out}``, ``{std_err}``, ``{job_id}``,
    ``{name}``, ``{user_name}``, ``{start_time_ns}``, ``{end_time_ns}``).
    For APPLOG, after substitution, if the path resolves to a directory, every
    regular file directly inside the directory is read (non-recursive); if it
    resolves to a file, that file is read. For SYSLOG, each path is a base
    directory containing one child directory per context node. Missing APPLOG
    paths are skipped silently, while missing SYSLOG node directories warn.
    A file whose name ends in ``.jsonl`` is read as one JSON entry per line
    and sent unchanged; for SYSLOG such a path names the file directly
    instead of a base directory.

    ``max_bytes_per_file`` defaults to :data:`DEFAULT_MAX_BYTES` (2 GB) so a
    runaway multi-GB training log can't OOM the blueprint. Larger files are
    capped to the **last** N bytes; smaller files are read in full. Pass an
    explicit value (int bytes or string like ``"500MB"``) to override, or
    ``None`` to disable the cap entirely. Tailing is correct here because
    crash symptoms (XID, NV_ERR_, NCCL aborts) typically appear near the end
    of the file rather than the beginning.
    """

    def __init__(
        self,
        *,
        log_path_templates: list[str],
        source: Source = Source.APPLOG,
        max_bytes_per_file: int | str | None = DEFAULT_MAX_BYTES,
    ):
        if not log_path_templates:
            raise ValueError("local_file: at least one path template required")
        self._templates = list(log_path_templates)
        self._source = source
        self._max_bytes = parse_size(max_bytes_per_file) if max_bytes_per_file is not None else None

    @property
    def source(self) -> Source:
        return self._source

    def collect(self, context: ExecutionContext) -> list[Observation]:
        # Reject naive datetimes at the boundary; the service rejects them and
        # attribution_lib has paths that crash on them.
        assert_aware(context["start_time"], "start_time")
        assert_aware(context["end_time"], "end_time")
        if self._source == Source.SYSLOG:
            return self._collect_syslog(context)
        observations: list[Observation] = []
        seen: set[Path] = set()
        for template in self._templates:
            for path in self._resolve_paths(template, context):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                observation = self._file_observation(path, context)
                if observation is not None:
                    observations.append(observation)
        return observations

    def _file_observation(self, path: Path, context: ExecutionContext) -> Optional[Observation]:
        """Read one file into an observation, or ``None`` when there is nothing to send."""
        # Distinguish "file does not exist" (silent skip) from
        # "file exists but we can't read it" (warn). `path.is_file()`
        # alone collapses both EACCES and ENOENT to False; check
        # explicitly so permission-denied attempts are visible in logs
        # (and downstream eval filters can exclude them deliberately).
        try:
            is_file = path.is_file()
        except PermissionError as e:
            logger.warning(f"local_file: permission denied stat'ing {path} for ctx {context.get('id')}: {e}")
            return None
        if not is_file:
            if path.exists():
                logger.warning(f"local_file: {path} exists but is not a regular file (ctx {context.get('id')})")
            else:
                logger.debug(f"local_file: no file at {path} for ctx {context.get('id')}")
            return None
        try:
            body = self._read(path)
        except PermissionError as e:
            logger.warning(f"local_file: permission denied reading {path} for ctx {context.get('id')}: {e}")
            return None
        if not body:
            return None

        if path.suffix == ".jsonl":
            entries = _jsonl_entries(path, body, context)
            if not entries:
                return None
        else:
            entries = entries_from_text(body)

        return log_observation(source=self._source, entries=entries, context=context)

    def _collect_syslog(self, context: ExecutionContext) -> list[Observation]:
        """Collect static syslog files from one child directory per context node.

        Args:
            context: Execution context whose ``nodes`` identify required syslog
                child directories below each configured base path.

        Returns:
            One observation per readable, non-empty syslog file.
        """
        observations: list[Observation] = []
        seen: set[Path] = set()
        found_syslog_input = False
        for template in self._templates:
            rendered = self._render_template(template, context)
            if rendered is None:
                continue
            base = Path(rendered)
            if base.suffix == ".jsonl":
                resolved = base.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                found_syslog_input = True
                try:
                    is_file = base.is_file()
                except PermissionError:
                    is_file = True  # _file_observation reports the denial
                if not is_file:
                    logger.warning(
                        f"local_file SYSLOG: expected syslog file {base} for ctx {context.get('id')}; skipping"
                    )
                    continue
                observation = self._file_observation(base, context)
                if observation is not None:
                    observations.append(observation)
                continue
            try:
                base_is_dir = base.is_dir()
            except PermissionError as e:
                logger.warning(f"local_file SYSLOG: permission denied stat'ing {base} for ctx {context.get('id')}: {e}")
                continue
            if base_is_dir is False:
                logger.warning(
                    f"local_file SYSLOG: expected syslog base directory {base} for ctx {context.get('id')}; skipping"
                )
                continue

            for node in context["nodes"]:
                node_dir = base / node
                try:
                    node_is_dir = node_dir.is_dir()
                except PermissionError as e:
                    logger.warning(
                        f"local_file SYSLOG: permission denied stat'ing {node_dir} "
                        f"for node {node} ctx {context.get('id')}: {e}"
                    )
                    continue
                if node_is_dir is False:
                    logger.debug(
                        f"local_file SYSLOG: missing syslog directory {node_dir} "
                        f"for node {node} ctx {context.get('id')}"
                    )
                    continue
                found_syslog_input = True
                try:
                    paths = sorted(p for p in node_dir.iterdir() if p.is_file())
                except PermissionError as e:
                    logger.warning(
                        f"local_file SYSLOG: permission denied listing {node_dir} "
                        f"for node {node} ctx {context.get('id')}: {e}"
                    )
                    continue

                for path in paths:
                    resolved = path.resolve()
                    if resolved in seen:
                        continue
                    if path.suffix == ".jsonl":
                        # Node directories hold raw logs; pre-shaped entries are configured
                        # as the path itself. Parsing this as raw syslog would drop every line.
                        # Not recorded in `seen`, so the same file still reads when it is also
                        # configured as a path, in either order.
                        logger.warning(
                            f"local_file SYSLOG: skipping {path}, a .jsonl file is read only when "
                            f"configured as the path itself (ctx {context.get('id')})"
                        )
                        continue
                    seen.add(resolved)
                    try:
                        body = self._read(path)
                    except PermissionError as e:
                        logger.warning(
                            f"local_file SYSLOG: permission denied reading {path} "
                            f"for node {node} ctx {context.get('id')}: {e}"
                        )
                        continue
                    if body == "":
                        continue
                    entries = syslog_entries_from_text(body, context, node=node)
                    observations.append(log_observation(source=self._source, entries=entries, context=context))
        if found_syslog_input is False:
            logger.warning(f"local_file SYSLOG: no syslog directories found for ctx {context.get('id')}")
        return observations

    def _resolve_paths(self, template: str, context: ExecutionContext) -> list[Path]:
        # `{std_out}` / `{std_err}` are pre-rendered by each ECA — the
        # Slurm ECA, for instance, resolves `%j`/`%x`/`${SLURM_*}` before
        # writing these to the context (see
        # `connectors/execution_contexts/slurm/filename_patterns.py`),
        # so by the time we see them they're fully-resolved filesystem
        # paths. The local_file connector therefore only handles generic
        # Python `.format()` placeholders against cross-cutting
        # `ExecutionContext` fields. Adding new placeholders here means
        # extending the cross-cutting `ExecutionContext` TypedDict, not
        # baking scheduler-specific token syntax into this file.
        rendered = self._render_template(template, context)
        if rendered is None:
            return []
        path = Path(rendered)
        # is_dir() can raise PermissionError on filesystems where a parent
        # dir is accessible but the leaf isn't (common with Lustre user dirs).
        # Treat that as "no file we can read here" and return empty.
        try:
            is_dir = path.is_dir()
        except PermissionError as e:
            logger.warning(f"local_file: permission denied stat'ing {path} for ctx {context.get('id')}: {e}")
            return []
        if is_dir:
            try:
                children = sorted(p for p in path.iterdir() if p.is_file())
            except PermissionError as e:
                logger.warning(f"local_file: permission denied listing {path} for ctx {context.get('id')}: {e}")
                return []
            return children
        return [path]

    def _render_template(self, template: str, context: ExecutionContext) -> str | None:
        """Render one local-file path template against an execution context.

        Args:
            template: Python-format path template from connector config.
            context: Execution context that supplies supported placeholders.

        Returns:
            Rendered path text, or ``None`` when the template is invalid or
            resolves to an empty path.
        """
        std_out = context.get("std_out")
        if std_out is None:
            std_out = ""
        std_err = context.get("std_err")
        if std_err is None:
            std_err = ""
        job_id = context.get("job_id")
        if job_id is None or job_id == "":
            job_id = context.get("id", "")
        name = context.get("name")
        if name is None:
            name = ""
        user_name = context.get("user_name")
        if user_name is None:
            user_name = ""
        try:
            rendered = template.format(
                std_out=std_out,
                std_err=std_err,
                job_id=job_id,
                name=name,
                user_name=user_name,
                start_time_ns=int(context["start_time"].timestamp() * 1e9),
                end_time_ns=int(context["end_time"].timestamp() * 1e9),
            )
        except KeyError as e:
            logger.warning(f"local_file template missing kwarg {e}; skipping")
            return None
        if rendered == "":
            return None
        return rendered

    def _read(self, path: Path) -> str:
        with path.open("rb") as fh:
            if self._max_bytes is None:
                data = fh.read()
            else:
                filesize = path.stat().st_size
                if filesize > self._max_bytes:
                    logger.debug(
                        f"local_file: {path} is {filesize}B > cap {self._max_bytes}B, reading last {self._max_bytes}B"
                    )
                    start = filesize - self._max_bytes
                    if path.suffix == ".jsonl":
                        # The tail usually starts mid-record, and that fragment would fail JSON
                        # parsing and skip the whole file, so drop everything up to the first
                        # newline. Starting one byte early makes an aligned tail cut only that
                        # newline. Trimming the capped buffer keeps memory bounded.
                        fh.seek(start - 1)
                        data = fh.read()
                        cut = data.find(b"\n")
                        data = data[cut + 1 :] if cut >= 0 else b""
                    else:
                        fh.seek(start)
                        data = fh.read()
                else:
                    data = fh.read()
        return data.decode("utf-8", errors="replace")


from attribution_blueprint import mca_registry as _registry  # noqa: E402
from attribution_blueprint.mca_registry import _PARAM  # noqa: E402
from attribution_blueprint.types import Source as _Source  # noqa: E402


def _build(spec: dict, ctx: dict) -> "LocalFileLogConnector":
    source: _Source = ctx["source"]
    paths = spec.get("paths")
    if not paths:
        raise MCAError(f"log.{source.name.lower()}.local_file requires paths")
    if isinstance(paths, str):
        paths = [paths]
    kwargs: dict = {"log_path_templates": list(paths), "source": source}
    # Only forward max_bytes if explicitly set so the constructor default
    # (DEFAULT_MAX_BYTES) applies for omitted config.
    if "max_bytes" in spec:
        kwargs["max_bytes_per_file"] = spec["max_bytes"]
    return LocalFileLogConnector(**kwargs)


_registry.register_component(
    framework="log",
    name="local_file",
    schema={"paths": _PARAM, "max_bytes": _PARAM},
    build=_build,
)


def _jsonl_entries(path: Path, body: str, context: ExecutionContext) -> Optional[list[TimeseriesEntry]]:
    """Parse one JSON value per non-blank line, or `None` when any line is not JSON."""
    entries: list[TimeseriesEntry] = []
    # Not splitlines(): raw U+2028/U+2029 are legal inside JSON strings and would split a line.
    for line_number, line in enumerate(body.split("\n"), start=1):
        if line.strip() == "":
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning(
                f"local_file: skipping {path}, line {line_number} is not valid JSON (ctx {context.get('id')})"
            )
            return None
    return entries


__all__ = ["LocalFileLogConnector", "DEFAULT_MAX_BYTES", "parse_size"]
