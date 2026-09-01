# syntax=docker/dockerfile:1.7
# Stage 1: build the React SPA
FROM node:22-bookworm-slim AS web-build
WORKDIR /frontend
COPY blueprint/src/attribution_blueprint/webui/frontend/ ./
RUN corepack enable && pnpm install --frozen-lockfile && pnpm exec tsc -b && pnpm exec vite build --outDir dist

# Stage 2: python app
FROM python:3.14.6-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:0.10.12 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV ATTRIBUTION_DB=/data/attributions.db
ENV ATTRIBUTION_SERVICE_PORT=8000
ENV ATTRIBUTION_WEB_PORT=8080

WORKDIR /opt/blueprint
COPY blueprint/pyproject.toml blueprint/uv.lock ./
COPY blueprint/src/ ./src/
# built SPA -> the package's static dir, BEFORE uv sync packages it
COPY --from=web-build /frontend/dist ./src/attribution_blueprint/webui/static/
COPY service/service-app/nvidia_attribution_client/ /opt/service/service-app/nvidia_attribution_client/
RUN uv sync --extra web --no-dev --no-editable --frozen

WORKDIR /opt/attribution-service
# attribution-lib is bundled at service/attribution-lib and the service
# pyproject pins it at ../attribution-lib, so it lands beside the service
# workdir at /opt/attribution-lib.
COPY service/attribution-lib/ /opt/attribution-lib/
COPY service/service-app/pyproject.toml service/service-app/uv.lock ./
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
