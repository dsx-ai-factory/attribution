# syntax=docker/dockerfile:1.7
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM python:3.14.6-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:0.10.12 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /opt/blueprint
COPY blueprint/pyproject.toml blueprint/uv.lock ./
COPY blueprint/src/ ./src/
COPY service/service-app/nvidia_attribution_client/ /opt/service/service-app/nvidia_attribution_client/
RUN uv sync --extra pod-placement-recorder --no-dev --no-editable --frozen

ENTRYPOINT ["/opt/blueprint/.venv/bin/pod-placement-recorder"]
