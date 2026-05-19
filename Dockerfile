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
 && apt-get install -y --no-install-recommends ca-certificates tini \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for layer caching. requirements.txt is generated
# from uv.lock by scripts/dev/sync_requirements.py.
COPY LICENSE ./LICENSE
COPY packages/sdk ./packages/sdk
COPY packages/hook ./packages/hook
COPY src/lumilake_server/requirements.txt ./requirements.txt
RUN pip install ./packages/sdk ./packages/hook \
 && pip install -r requirements.txt

# Image-only server code. Not published to PyPI.
COPY src/lumilake_server ./lumilake_server

RUN mkdir -p /app/plugins

# uid/gid 10001 keeps the in-container user out of the host UID range.
RUN groupadd --gid 10001 lumilake \
 && useradd --no-create-home --uid 10001 --gid 10001 --shell /usr/sbin/nologin lumilake \
 && chown -R lumilake:lumilake /app

ENV PYTHONPATH=/app:/app/plugins
USER lumilake

EXPOSE 9000

ARG BUILD_VERSION=dev
ARG BUILD_REF=local
ARG BUILD_CREATED=unknown
LABEL org.opencontainers.image.version="${BUILD_VERSION}" \
      org.opencontainers.image.created="${BUILD_CREATED}" \
      org.opencontainers.image.revision="${BUILD_REF}"

# tini as PID 1 reaps zombies and forwards SIGTERM to uvicorn.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "lumilake_server.main"]
