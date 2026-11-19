FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra music-separation --extra youtube-publishing

COPY . .

ENV WEB_HOST=0.0.0.0 \
    WEB_PORT=8000 \
    STREAMCLIPPER_DATA_DIR=/app/data \
    LOCALAPPDATA=/app/data

EXPOSE 8000
VOLUME ["/app/data"]

CMD ["uv", "run", "--no-sync", "python", "live_clipper.py"]
