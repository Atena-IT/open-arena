# ---------------------------------------------------------------------------
# Stage 1 -- builder
# Install uv and resolve/install all project dependencies into /app/.venv.
# We copy only the files that affect the dependency install first so that
# Docker's layer cache is reused on code-only changes.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy the workspace manifests first (cache-friendly layer).
# The workspace root pyproject.toml + uv.lock define all transitive deps;
# the sub-packages under packages/ are local workspace members.
COPY pyproject.toml uv.lock ./
COPY packages/open-arena-core/pyproject.toml ./packages/open-arena-core/
COPY packages/open-arena-cli/pyproject.toml ./packages/open-arena-cli/

# Install all non-dev dependencies for the entire workspace into /app/.venv.
# --frozen: use uv.lock as-is; --no-dev: skip dev extras.
RUN uv sync --frozen --no-dev

# ---------------------------------------------------------------------------
# Stage 2 -- runtime
# Copy the venv from builder, copy source code, add a non-root user.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Create a non-root user for running the service
RUN groupadd --gid 1001 arena && \
    useradd --uid 1001 --gid arena --shell /bin/bash --create-home arena

WORKDIR /app

# Copy the pre-built virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy the workspace source and assets
COPY src/ ./src/
COPY packages/open-arena-core/ ./packages/open-arena-core/
COPY packages/open-arena-cli/ ./packages/open-arena-cli/
COPY pyproject.toml ./

# Ensure the arena state directory exists and is writable by the arena user
RUN mkdir -p /app/.open-arena && chown -R arena:arena /app/.open-arena && \
    chown -R arena:arena /app

# Activate the venv by prepending it to PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Switch to non-root user
USER arena

# Arena state dir: default is .open-arena/ relative to cwd.
# The docker-compose volume mounts over this path so state persists.
VOLUME ["/app/.open-arena"]

EXPOSE 8000

# Default command: start the API server.
# The `arena` script is provided by the open-arena-cli workspace package,
# which is installed as part of the full `open-arena` workspace install.
# Override at runtime with a different `arena` sub-command.
CMD ["arena", "serve", "--host", "0.0.0.0", "--port", "8000"]
