# syntax=docker/dockerfile:1
#
# agent-base runtime image.
# Multi-stage: the builder resolves and installs the project (plus the
# migrate extra needed for alembic upgrade head) into /install;
# the runtime stage carries only the installed site-packages, alembic
# assets and a non-root user.

FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Metadata + package sources only: tests, docs and local state stay out.
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src

RUN pip install --prefix=/install ".[migrate]"


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Dedicated non-root identity (uid/gid 10001).
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --no-create-home \
       --shell /usr/sbin/nologin app

COPY --from=builder /install/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /install/bin /usr/local/bin

WORKDIR /app

# Migration assets are required at runtime (compose runs alembic upgrade head).
COPY alembic.ini ./
COPY alembic ./alembic

# Writable location for the sqlite fallback checkpointer / uploads.
RUN mkdir -p /app/data && chown -R app:app /app

USER app

EXPOSE 8000

# Liveness probe: /health answers HTTP 200 even when degraded.
# Readiness should additionally parse the body and require status == ok.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"]

CMD ["python", "-m", "uvicorn", "agent_base.entrypoints.server:app", "--host", "0.0.0.0", "--port", "8000"]
