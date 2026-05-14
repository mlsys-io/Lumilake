# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.title="Lumilake Server" \
      org.opencontainers.image.description="Lumilake server runtime" \
      org.opencontainers.image.source="https://github.com/mlsys-io/lumilake_OSS" \
      org.opencontainers.image.url="https://github.com/mlsys-io/lumilake_OSS"

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the dependency closure first for layer caching. The server
# image installs the published lumilake-sdk interface wheel (which
# carries the shared envs.py + log.py) plus a flat pin file derived
# from uv.lock (scripts/dev/sync_requirements.py).
COPY packages/sdk ./packages/sdk
COPY src/lumilake_server/requirements.txt ./requirements.txt
RUN pip install ./packages/sdk \
 && pip install -r requirements.txt

# Image-only server code. Not published to PyPI.
COPY src/lumilake_server ./lumilake_server

ENV PYTHONPATH=/app

EXPOSE 9000

ARG BUILD_VERSION=dev
ARG BUILD_REF=local
ARG BUILD_CREATED=unknown
LABEL org.opencontainers.image.version="${BUILD_VERSION}" \
      org.opencontainers.image.created="${BUILD_CREATED}" \
      org.opencontainers.image.revision="${BUILD_REF}"

# Shell form so ${VAR} interpolation happens at container start.
CMD python -m lumilake_server.main --host 0.0.0.0 --port "${LUMILAKE_SERVER_PORT:-9000}"
