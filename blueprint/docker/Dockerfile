# syntax=docker/dockerfile:1.7
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Stage 1: build the React SPA
FROM node:22-bookworm-slim AS web-build
WORKDIR /repo/blueprint/src/attribution_blueprint/webui/frontend
COPY THIRD_PARTY_NOTICES.md /repo/THIRD_PARTY_NOTICES.md
COPY blueprint/src/attribution_blueprint/webui/frontend/ ./
RUN corepack enable && pnpm install --frozen-lockfile && pnpm exec tsc -b && pnpm exec vite build --outDir dist

# Stage 2: python app
FROM python:3.14.6-slim-trixie
LABEL com.nvidia.attribution.project.license="Apache-2.0"
COPY --from=ghcr.io/astral-sh/uv:0.10.12 /uv /uvx /bin/
COPY LICENSE /licenses/LICENSE
COPY --from=web-build /repo/blueprint/src/attribution_blueprint/webui/frontend/dist/assets/THIRD_PARTY_NOTICES.txt /licenses/THIRD_PARTY_NOTICES.txt

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV ATTRIBUTION_DB=/data/attributions.db
ENV ATTRIBUTION_SERVICE_PORT=8000
ENV ATTRIBUTION_WEB_PORT=8080

WORKDIR /opt/blueprint
COPY blueprint/pyproject.toml blueprint/uv.lock blueprint/LICENSE ./
COPY blueprint/src/ ./src/
# built SPA -> the package's static dir, BEFORE uv sync packages it
COPY --from=web-build /repo/blueprint/src/attribution_blueprint/webui/frontend/dist ./src/attribution_blueprint/webui/static/
COPY service/service-app/nvidia_attribution_client/ /opt/service/service-app/nvidia_attribution_client/
RUN uv sync --extra web --no-dev --no-editable --frozen

WORKDIR /opt/attribution-service
# attribution-lib is bundled at service/attribution-lib and the service
# pyproject pins it at ../attribution-lib, so it lands beside the service
# workdir at /opt/attribution-lib.
COPY service/attribution-lib/ /opt/attribution-lib/
COPY service/service-app/pyproject.toml service/service-app/uv.lock ./
# hatchling reads the package version from this file at wheel-build time.
COPY service/service-app/_build_version.py ./
COPY service/service-app/README.md service/service-app/LICENSE ./
COPY service/service-app/app/ ./app/
COPY service/service-app/nvidia_attribution_client/ ./nvidia_attribution_client/
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && uv sync --no-dev --no-editable --frozen \
    && apt-get purge -y --auto-remove gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY blueprint/docker/start.sh /usr/local/bin/attribution-blueprint-start
RUN chmod +x /usr/local/bin/attribution-blueprint-start

WORKDIR /opt/blueprint
EXPOSE 8000 8080

CMD ["/usr/local/bin/attribution-blueprint-start"]
