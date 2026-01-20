# 1. Use the verified official tag (bookworm-slim)
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS base

# 2. Install build tools (essential for scientific packages on 3.13)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Copy only dependency files for better caching
COPY uv.lock pyproject.toml ./

# 4. Install dependencies (This will compile scikit-learn - please be patient!)
RUN uv sync --frozen --no-install-project

# 5. Copy the rest of the project
COPY . .

# 6. Run the training script
ENTRYPOINT ["uv", "run", "src/plants/train.py"]
