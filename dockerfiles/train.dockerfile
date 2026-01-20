FROM ghcr.io/astral-sh/uv:python3.13-bookworm AS base

WORKDIR /

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src/ src/
COPY data/ data/

RUN uv sync --frozen --no-install-project

ENTRYPOINT ["uv", "run", "src/plants/train.py"]
