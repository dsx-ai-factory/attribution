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

# Install uv
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1


# Change the working directory to the `app` directory
WORKDIR /app

# Install dependencies
COPY uv.lock pyproject.toml .
COPY service/attribution-lib/ ./service/attribution-lib/

#RUN --mount=type=cache,target=/root/.cache/uv \
#    --mount=type=bind,source=uv.lock,target=uv.lock \
#    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \

RUN  uv sync --frozen --no-install-project --no-editable --no-dev

# Copy the project into the intermediate image
ADD . /app


# Bumps the version

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable --no-dev

FROM python:3.12-slim

ENV PATH="/app/.venv/bin:$PATH"

# Copy the environment, but not the source code
COPY --from=builder /app/.venv /app/.venv

EXPOSE 8000

# Run the application
CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
