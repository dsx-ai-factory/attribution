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
