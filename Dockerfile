# syntax=docker/dockerfile:1
# Digest-pinned official multi-arch uv image, used only at build time via
# bind mounts so no build tool lands in the final image. Keep the version in
# step with [tool.uv].required-version and the setup-uv `version:` in CI.
FROM ghcr.io/astral-sh/uv:0.13.5@sha256:32a7bb39b496babbd997b0a39a4d66c126bdaf5661b302bf6f06abda8deca6ff AS uv

FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.title="Lumilake Server" \
      org.opencontainers.image.description="Lumilake server runtime" \
      org.opencontainers.image.source="https://github.com/mlsys-io/lumilake_OSS" \
      org.opencontainers.image.url="https://github.com/mlsys-io/lumilake_OSS"

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends ca-certificates tini \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The image ships no pip: nothing in the server needs it, and the base
# image's seeded pip is old and carries a vendored CA bundle. Uninstall it
# and delete ensurepip's bundled pip wheel so no pip or pip-vendored certifi
# remains.
RUN --mount=from=uv,source=/uv,target=/bin/uv uv pip uninstall --system pip \
 && rm -rf /usr/local/lib/python3.12/ensurepip/_bundled

# Install deps first for layer caching. requirements.txt is generated
# from uv.lock by scripts/dev/sync_requirements.py.
COPY src/lumilake_server/requirements.txt ./requirements.txt
RUN --mount=from=uv,source=/uv,target=/bin/uv uv pip install --system -r requirements.txt

COPY LICENSE ./LICENSE
COPY packages/sdk ./packages/sdk
COPY packages/hook ./packages/hook
RUN --mount=from=uv,source=/uv,target=/bin/uv uv pip install --system --no-deps ./packages/sdk ./packages/hook

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
