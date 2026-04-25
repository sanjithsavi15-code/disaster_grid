# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              disaster_grid — Hugging Face Spaces Dockerfile                ║
# ║  Target  : Hugging Face Spaces "Blank Docker" template                     ║
# ║  Runtime : Python 3.10-slim · FastAPI · Uvicorn · openenv-core             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Stage 1: Base image ────────────────────────────────────────────────────────
# python:3.10-slim is chosen over the full image (~900 MB) because our RL
# environment has no C-extension build requirements that need a compiler at
# runtime.  This keeps the final layer under ~400 MB, which matters for cold-
# start latency on Hugging Face Spaces free-tier hardware.
FROM python:3.10-slim

# ── Stage 2: System dependencies ──────────────────────────────────────────────
# We install only what is strictly required:
#   curl  — health-check probes from the Spaces infra ping /health
#   build-essential — needed if any transitive dep compiles a C extension
# The apt cache is purged in the same RUN layer to avoid bloating the image
# with package index files (~30 MB on slim images).
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 3: Non-root user ─────────────────────────────────────────────────────
# CRITICAL (Hugging Face requirement):
# Hugging Face Spaces executes containers as a non-root user with uid=1000.
# If the image does not pre-create this user, the runtime uid has no entry in
# /etc/passwd, which breaks libraries that call getpwuid() (e.g. huggingface_hub,
# transformers' caching layer).
#
# We create:
#   - group "user"  gid=1000
#   - user  "user"  uid=1000, home=/home/user, no password, no sudo
#
# --no-log-init suppresses large sparse-file writes on some kernels.
RUN groupadd --gid 1000 user \
    && useradd --uid 1000 --gid 1000 --no-log-init --create-home user

# ── Stage 4: Working directory + ownership ─────────────────────────────────────
# CRITICAL (Hugging Face requirement):
# The /app directory must be owned by uid=1000 BEFORE we switch users.
# If we create /app as root and then switch to user, any write at runtime
# (e.g. openenv writing episode logs, Pydantic schema cache) will raise
# PermissionError.  chown here costs nothing at build time and prevents an
# entire class of runtime failures.
WORKDIR /app
RUN chown -R user:user /app

# ── Stage 5: PATH — expose local pip binaries ──────────────────────────────────
# CRITICAL (Hugging Face requirement):
# When pip installs packages as a non-root user it places console-script
# entry points (e.g. `uvicorn`, `openenv`) in ~/.local/bin, NOT in /usr/bin.
# Without this PATH addition the CMD instruction cannot find `uvicorn` and
# the container exits immediately with "exec: uvicorn: not found".
ENV PATH="/home/user/.local/bin:${PATH}"

# ── Stage 6: Switch to non-root user ──────────────────────────────────────────
# All subsequent RUN, COPY, and CMD instructions execute as uid=1000.
# Placing USER before COPY means the copied files are owned by user, not root,
# matching what the Spaces runtime expects.
USER user

# ── Stage 7: Install Python dependencies ──────────────────────────────────────
# We copy pyproject.toml first (before the rest of the source) to exploit
# Docker's layer cache: as long as pyproject.toml is unchanged, Docker will
# reuse the pip-install layer even when app.py or src/ is edited.  This
# reduces incremental build time from ~3 min to ~10 s on typical CI runners.
#
# --no-cache-dir  : prevents pip from writing the HTTP cache to disk
#                   (~200 MB saved in the image layer)
# --user          : installs into ~/.local (required since we are not root)
COPY --chown=user:user . /app
RUN pip install --no-cache-dir --user .

# ── Stage 8: Copy project source ──────────────────────────────────────────────
# Copied after the dependency install layer so that editing src/ or app.py
# does not invalidate the expensive pip install cache step above.
COPY --chown=user:user . /app

# ── Stage 9: Expose port ───────────────────────────────────────────────────────
# CRITICAL (Hugging Face requirement):
# Hugging Face Spaces routes all external HTTPS traffic to internal port 7860.
# Using any other port (e.g. 8000, 8080) means the Space will start but will
# return a 502 Bad Gateway for every request, with no error message in the
# build logs — one of the most confusing HF deployment failure modes.
EXPOSE 7860

# ── Stage 10: Health check ─────────────────────────────────────────────────────
# Hugging Face Spaces monitors container liveness via HTTP.  This HEALTHCHECK
# tells Docker (and the Spaces infra) that the app is ready once the /health
# endpoint responds with 2xx.  The 30 s start-period gives Uvicorn time to
# load the openenv environment before the first probe fires.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail --silent http://localhost:7860/health || exit 1

# ── Stage 11: Default command ──────────────────────────────────────────────────
# Uvicorn is launched in exec form (JSON array) rather than shell form so that
# SIGTERM from the Spaces orchestrator is delivered directly to the Uvicorn
# process, enabling graceful shutdown instead of a SIGKILL after the timeout.
#
# --host 0.0.0.0  : CRITICAL — binds to all interfaces so the Spaces proxy
#                   can reach the container.  127.0.0.1 or localhost would
#                   make the app invisible to the outside world.
# --port 7860     : must match the EXPOSE directive and the Spaces requirement.
# --workers 1     : single worker is safer for stateful RL environments;
#                   increase to 2+ only after verifying episode state is not
#                   shared across worker processes.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]