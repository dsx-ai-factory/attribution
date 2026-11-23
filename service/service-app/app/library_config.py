# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TOML file configuration for the service's attribution library."""

import csv
import os
import re
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import attribution_lib
from attribution_lib.core import setup_rules
from attribution_lib.observation.inbound import Source
from attribution_lib.rules_engine import SOURCE_CONFIGS, register_source_config
from loguru import logger
from pydantic import AfterValidator, BaseModel, ConfigDict, ValidationError, field_validator

CONFIG_FILE_VARIABLE = "ATTRIBUTION_CONFIG_FILE"


def _nonempty(value: str) -> str:
    """Reject a string that holds only whitespace.

    Args:
        value: String from the configuration file.

    Returns:
        The unchanged string.
    """
    if value.strip() == "":
        raise ValueError("must be a nonempty string")
    return value


NonEmptyString = Annotated[str, AfterValidator(_nonempty)]


class LibrarySettings(BaseModel):
    """Attribution library settings. An unset key keeps the library default."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    applog_parse_timestamp: bool | None = None
    ignored_hostname_prefixes: list[NonEmptyString] | None = None
    node_mapping_file: NonEmptyString | None = None
    node_domain_regex: str | None = None
    rules_paths: list[NonEmptyString] | None = None

    @field_validator("node_domain_regex")
    @classmethod
    def _check_domain_regex(cls, value: str | None) -> str | None:
        """Require a valid regex with a named 'domain' group.

        Args:
            value: Regex from the configuration file.

        Returns:
            The unchanged regex.
        """
        if value is None:
            return None
        message = "must be a valid regex with a named 'domain' group"
        try:
            pattern = re.compile(value, re.IGNORECASE)
        except re.PatternError:
            raise ValueError(message) from None
        if "domain" not in pattern.groupindex:
            raise ValueError(message)
        return value


def load_settings() -> tuple[LibrarySettings, Path]:
    """Read library settings from the TOML file that ATTRIBUTION_CONFIG_FILE names.

    Returns:
        The settings, and the folder that relative paths in them resolve from.
        When the variable is unset, the default settings and the working folder.
    """
    raw = os.getenv(CONFIG_FILE_VARIABLE)
    if raw is None:
        return LibrarySettings(), Path()
    path = Path(raw)
    try:
        with path.open("rb") as stream:
            settings = LibrarySettings.model_validate(tomllib.load(stream))
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as error:
        raise ValueError(f"{CONFIG_FILE_VARIABLE} file {path} is invalid: {error}") from None
    return settings, path.parent


def _node_domain_resolver(regex: str) -> attribution_lib.NodeDomainResolver:
    """Build a hostname resolver from a validated domain regex.

    Args:
        regex: Regex with a named 'domain' group.

    Returns:
        Function that maps a hostname to its domain.
    """
    pattern = re.compile(regex, re.IGNORECASE)

    def resolve(hostname: str) -> str | None:
        """Return the lowercase domain of a hostname, or None for no match.

        Args:
            hostname: Node hostname, optionally with a DNS suffix.

        Returns:
            Matched domain, or None when the hostname has no domain.
        """
        match = pattern.search(hostname.lower().split(".", 1)[0])
        return match.group("domain") if match is not None else None

    return resolve


def configure_library() -> None:
    """Initialize library rules and APPLOG timestamp parsing from the configuration file."""
    settings, base = load_settings()
    prefixes = settings.ignored_hostname_prefixes
    mapping_file = str(base / settings.node_mapping_file) if settings.node_mapping_file is not None else None
    resolver = _node_domain_resolver(settings.node_domain_regex) if settings.node_domain_regex is not None else None
    rules_paths = [base / path for path in settings.rules_paths] if settings.rules_paths is not None else None
    if rules_paths is not None and any(not path.is_dir() for path in rules_paths):
        raise ValueError(f"{CONFIG_FILE_VARIABLE} rules_paths must contain directory paths that exist")

    setup_rules(
        node_mapping_file=mapping_file,
        ignored_hostname_prefixes=prefixes,
        node_domain_resolver=resolver,
        rules_paths=rules_paths,
        force=any(value is not None for value in (prefixes, mapping_file, resolver, rules_paths)),
    )
    if rules_paths is not None and attribution_lib.get_rules_engine().all_rule_names == set():
        logger.warning("rules_paths loaded no attribution rules")
    if mapping_file is not None:
        with Path(mapping_file).open(newline="") as stream:
            columns = csv.DictReader(stream).fieldnames
        if columns is None or "instanceId" not in columns:
            logger.warning("node_mapping_file has no instanceId column. Instance ID lookup is unavailable.")
    # setup_rules treats an empty list as a request to preserve the current prefixes.
    if prefixes == []:
        attribution_lib.ignored_hostname_prefixes = []
    if settings.applog_parse_timestamp is not None:
        config = SOURCE_CONFIGS[Source.APPLOG]
        register_source_config(
            Source.APPLOG,
            replace(
                config, body_fallbacks=replace(config.body_fallbacks, parse_timestamp=settings.applog_parse_timestamp)
            ),
        )
