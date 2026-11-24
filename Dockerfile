# Server image: runtime dependencies only (no dev tools, no content libraries such as Pillow and
# PyAV), run as a non-root user. Compose lives in the platform repo and passes every setting as an
# environment variable.

# Base images are pinned by tag and digest. Refresh both together.
FROM ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 AS uv

FROM python:3.12-slim@sha256:37fc60d303cb9294d3718b5b48998a0210bb1c73611c1399ef5eb7964c0536e5 AS build
COPY --from=uv /uv /bin/uv
# Use the image's Python, so the venv built here runs in the final stage (same base).
ENV UV_PYTHON_DOWNLOADS=0 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
# The project is not a package (`package = false`), so this installs the dependencies only.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-default-groups

FROM python:3.12-slim@sha256:37fc60d303cb9294d3718b5b48998a0210bb1c73611c1399ef5eb7964c0536e5
ENV PATH="/app/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=build /app/.venv ./.venv
COPY app ./app
COPY alembic.ini ./
# scripts/ is here so the puzzle buffer check can run on the server (check_puzzle_buffer.py).
COPY scripts ./scripts
# The mount point for local clue media. The server only reads it.
RUN mkdir -p var/media
USER app
EXPOSE 8000
# Compose's `migrate` service overrides this with `alembic upgrade head`.
# --forwarded-allow-ips '*' is safe only because the port is not published and only the proxy
# shares the network.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", \
     "--proxy-headers", "--forwarded-allow-ips", "*", "--no-server-header", \
     "--timeout-graceful-shutdown", "20"]
