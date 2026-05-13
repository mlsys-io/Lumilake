# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app
RUN pip install ".[server]"

EXPOSE 9000
# Honor LUMILAKE_SERVER_PORT from the env_file (default 9000). Shell form
# so ${VAR} interpolation happens at container start.
CMD python -m lumilake.server.main --host 0.0.0.0 --port "${LUMILAKE_SERVER_PORT:-9000}"
