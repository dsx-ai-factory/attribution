# syntax=docker/dockerfile:1.7
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Runtime base image. Override with an air-gapped mirror via --build-arg.
ARG BASE_IMAGE=python:3.14.6-slim-trixie

# Install uv
FROM python:3.14.6-slim-trixie AS builder
COPY --from=ghcr.io/astral-sh/uv:0.10.12 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1


RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# Change the working directory to the `app` directory
WORKDIR /app

# Install dependencies
COPY service/service-app/uv.lock service/service-app/pyproject.toml ./
COPY service/service-app/nvidia_attribution_client/ ./nvidia_attribution_client/
COPY service/attribution-lib/ /attribution-lib/

RUN  uv sync --frozen --no-install-project --no-editable --no-dev

# Copy the project into the intermediate image
COPY service/service-app/ /app/



# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable --no-dev

FROM ${BASE_IMAGE}

ENV PATH="/app/.venv/bin:$PATH"

# Copy the environment, but not the source code
COPY --from=builder /app/.venv /app/.venv

EXPOSE 8000

# Run the application
CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
