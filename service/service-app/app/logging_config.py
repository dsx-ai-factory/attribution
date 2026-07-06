# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import sys
from typing import Optional

from loguru import logger
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, LogRecordExporter
from opentelemetry.sdk.resources import Resource

_SERVICE_NAME = "attribution-service"
_provider: Optional[LoggerProvider] = None  # global, per-process singleton


def setup_logging() -> Optional[LoggerProvider]:
    """Configure all loguru sinks from scratch: stdout always, OTLP when configured.

    Clears every sink and re-adds them all here.

    Returns
        The OTLP provider (so the caller can flush/shutdown it) or None for stdout-only.
    """
    level = _resolve_log_level()
    logger.remove()
    logger.add(
        sys.stdout,
        level=level,
        format="{time:HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
    )

    provider = _get_or_create_provider()
    if provider is not None:
        logger.add(LoggingHandler(logger_provider=provider), level=level, format="{message}")

    return provider


def _resolve_log_level() -> str:
    """Resolve LOG_LEVEL env var to loguru level name, falling back to INFO on anything it rejects."""
    default_level = "INFO"
    level = os.environ.get("LOG_LEVEL", default_level).strip().upper()
    try:
        return logger.level(level).name
    except ValueError:
        return default_level


def _build_exporter(endpoint: str, token: Optional[str]) -> LogRecordExporter:
    """Build an OTLP gRPC LogRecordExporter.

    token is optional: when present, it adds a Bearer header.
    """
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

    headers = (("authorization", f"Bearer {token}"),) if token is not None else None
    return OTLPLogExporter(endpoint=endpoint, headers=headers)


def _get_or_create_provider() -> Optional[LoggerProvider]:
    """Return the process-wide OTLP LoggerProvider or None if unconfigured.

    Gated on OTEL_EXPORTER_OTLP_ENDPOINT (None when unset -> stdout-only).
    OTEL_EXPORTER_OTLP_TOKEN is optional: when present, it adds a Bearer header.
    """
    global _provider
    if _provider is not None:
        return _provider

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint is None:
        return None

    protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL")
    if protocol is not None and protocol.strip().lower() != "grpc":
        logger.warning(
            f"OTEL_EXPORTER_OTLP_PROTOCOL={protocol!r} is set but this service exports OTLP over gRPC only; ignoring it."
        )

    exporter = _build_exporter(endpoint, os.getenv("OTEL_EXPORTER_OTLP_TOKEN"))
    provider = LoggerProvider(resource=Resource.create({"service.name": _SERVICE_NAME}))
    provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
    _provider = provider
    return provider
