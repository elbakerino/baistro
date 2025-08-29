FROM python:3.10-slim-bookworm AS base

LABEL org.opencontainers.image.source = "https://github.com/elbakerino/baistro"
LABEL org.opencontainers.image.authors = "Michael Becker, https://i-am-digital.eu"
LABEL org.opencontainers.image.title = "baistro"
LABEL org.opencontainers.image.version = "0.1.0"
LABEL org.opencontainers.image.licenses = "MIT"

ENV PYTHONUNBUFFERED 1
ARG DEBIAN_FRONTEND=noninteractive

RUN pip install --no-cache-dir -Iv poetry

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-numpy \
    cargo \
    libssl-dev && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

RUN mkdir libs

COPY ./README.md README.md
COPY ./LICENSE LICENSE

FROM base AS dev

CMD poetry lock --no-interaction && poetry sync --no-interaction && poetry install --no-interaction && exec poetry run flask --app baistro.server:app --debug run --host 0.0.0.0 --port ${PORT}

FROM base

COPY ./pyproject.toml /app/pyproject.toml
COPY ./poetry.lock /app/poetry.lock

# todo: using the mount cache increases the image size
#RUN --mount=type=cache,target=/root/.cache/poetry poetry install --sync --no-dev --no-interaction -E gunicorn
RUN poetry install --sync --without dev --no-interaction --no-cache -E gunicorn

COPY ./baistro /app/baistro

ENV GUN_W 2

CMD exec poetry run gunicorn -w ${GUN_W} baistro.server:app
