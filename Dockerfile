# ─────────────────────────────────────────────────────────────
# Mitrailleuse gRPC server – production image
# ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS base

# locale & build utils
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------
# Layer 1: install Python deps
# -------------------------------------------------------------
FROM base AS builder
WORKDIR /install

# copy only the requirement spec first (build cache friendly)
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# -------------------------------------------------------------
# Layer 2: runtime image – copy app & deps
# -------------------------------------------------------------
FROM base AS runtime
WORKDIR /app

# copy installed packages from builder
COPY --from=builder /root/.local /usr/local

# copy source code
COPY mitrailleuse/ ./mitrailleuse/
COPY server.py .
COPY mitrailleuse.proto .

# health + reflection need no extra binaries; internal port
EXPOSE 50051

ENTRYPOINT ["python", "server.py"]