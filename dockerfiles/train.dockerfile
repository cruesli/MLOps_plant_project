# 1. Use the correct, verified slim tag
FROM ghcr.io/astral-sh/uv:3.13-slim AS base

# 2. Install build tools for scikit-learn (needed for compilation on 3.13)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Copy and install dependencies
COPY uv.lock pyproject.toml ./
RUN uv sync --frozen --no-install-project

# 4. Copy the rest of the project
COPY . .

# 5. Entrypoint
ENTRYPOINT ["uv", "run", "src/plants/train.py"]