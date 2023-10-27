FROM python:3.10-slim-bookworm AS base

ENV PYTHONUNBUFFERED 1
ARG DEBIAN_FRONTEND=noninteractive

RUN pip install --no-cache-dir -Iv poetry==1.3.2

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

CMD poetry lock --no-interaction --no-update && poetry install --sync --no-interaction && poetry run flask --app baistro.server:app --debug run --host 0.0.0.0 --port ${PORT}

FROM base

COPY ./pyproject.toml /app/pyproject.toml
COPY ./poetry.lock /app/poetry.lock

# todo: using the mount cache increases the image size
#RUN --mount=type=cache,target=/root/.cache/poetry poetry install --sync --no-dev --no-interaction -E gunicorn
RUN poetry install --sync --no-dev --no-interaction --no-cache -E gunicorn

COPY ./baistro /app/baistro

ENV GUN_W 2

CMD poetry run gunicorn -w ${GUN_W} baistro.server:app
