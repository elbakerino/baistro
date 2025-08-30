FROM python:3.12-slim-bookworm AS base

ARG VCS_REF
ARG VCS_URL
ARG CI_RUN_URL
ARG BUILD_DATE
ARG VERSION

LABEL org.opencontainers.image.source="https://github.com/elbakerino/baistro"
LABEL org.opencontainers.image.authors="Michael Becker, https://i-am-digital.eu"
LABEL org.opencontainers.image.title="baistro"
LABEL org.opencontainers.image.version="0.2.0"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.source=$VCS_URL
LABEL org.opencontainers.image.revision=$VCS_REF
LABEL org.opencontainers.image.url=$CI_RUN_URL
LABEL org.opencontainers.image.created=$BUILD_DATE
LABEL org.opencontainers.image.version=$VERSION

ENV PYTHONUNBUFFERED=1
# note: let's try adding the bytecode. the con-arguments are somewhat fuzzy and not backed by arguments
# (con) https://stackoverflow.com/a/60797635/2073149
# (pro) https://aleksac.me/blog/dont-use-pythondontwritebytecode-in-your-dockerfiles/
# \
#    PYTHONDONTWRITEBYTECODE=1

ENV POETRY_VERSION=2.1.4 \
    POETRY_NO_INTERACTION=1
    #\
    #POETRY_VIRTUALENVS_CREATE=false
ARG DEBIAN_FRONTEND=noninteractive

# no known bugs naymore, but locking poetry due to so many issues with minor updates in other packages
RUN pip install --no-cache-dir -Iv "poetry==${POETRY_VERSION}"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cargo \
    libssl-dev && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

FROM base AS dev

# note: with `poetry sync`/`--sync` lead to `poetry not found` error; WHEN `POETRY_VIRTUALENVS_CREATE=false` is set
#CMD poetry lock --no-interaction && poetry install --no-interaction && exec poetry run flask --app baistro.server:app --debug run --host 0.0.0.0 --port ${PORT}
CMD poetry lock --no-interaction && poetry sync --no-interaction && poetry install --no-interaction && exec poetry run flask --app baistro.server:app --debug run --host 0.0.0.0 --port ${PORT}

FROM base as builder

ENV POETRY_VIRTUALENVS_CREATE=false

COPY ./pyproject.toml /app/pyproject.toml
COPY ./poetry.lock /app/poetry.lock

# todo: using the mount cache increases the image size
#RUN --mount=type=cache,target=/root/.cache/poetry poetry install --sync --no-dev --no-interaction -E gunicorn
RUN poetry install --without dev -E gunicorn --no-cache --no-root --no-interaction
# note: `sync` doesn't work without venv (acording to docs "works not well")
#RUN poetry sync --without dev -E gunicorn --no-cache --no-root --no-interaction
# note: with `--compile` the image was ~150MB larger, no measurable performance gain (or unknown how to check)
#--compile

FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

#COPY --from=builder /usr/local /usr/local
#COPY --from=builder /root/.cache/pypoetry/virtualenvs /root/.cache/pypoetry/virtualenvs

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY ./pyproject.toml /app/pyproject.toml
COPY ./poetry.lock /app/poetry.lock

COPY ./baistro /app/baistro

COPY ./README.md README.md
COPY ./LICENSE LICENSE

ENV GUN_W 2

# note: kept gettings `command not found: gunicorn`, this was caused by the `/root/.cache` mounts in docker-compose.
#       remove these `volumes` when testing prod image locally!
CMD exec poetry run gunicorn -w ${GUN_W} baistro.server:app
