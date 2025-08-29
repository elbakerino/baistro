FROM python:3.10-slim-bookworm AS base

ARG VCS_REF
ARG VCS_URL
ARG CI_RUN_URL
ARG BUILD_DATE
ARG VERSION

LABEL org.opencontainers.image.source="https://github.com/elbakerino/baistro"
LABEL org.opencontainers.image.authors="Michael Becker, https://i-am-digital.eu"
LABEL org.opencontainers.image.title="baistro"
LABEL org.opencontainers.image.version="0.1.0"
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

RUN mkdir libs

COPY ./README.md README.md
COPY ./LICENSE LICENSE

FROM base AS dev

# note: with `poetry sync`/`--sync` lead to `poetry not found` error; WHEN `POETRY_VIRTUALENVS_CREATE=false` is set
#CMD poetry lock --no-interaction && poetry install --no-interaction && exec poetry run flask --app baistro.server:app --debug run --host 0.0.0.0 --port ${PORT}
CMD poetry lock --no-interaction && poetry sync --no-interaction && poetry install --no-interaction && exec poetry run flask --app baistro.server:app --debug run --host 0.0.0.0 --port ${PORT}

FROM base

COPY ./pyproject.toml /app/pyproject.toml
COPY ./poetry.lock /app/poetry.lock

# todo: using the mount cache increases the image size
#RUN --mount=type=cache,target=/root/.cache/poetry poetry install --sync --no-dev --no-interaction -E gunicorn
RUN poetry sync --without dev -E gunicorn --no-cache --no-root --no-interaction
#--compile

COPY ./baistro /app/baistro

ENV GUN_W 2

# note: kept gettings `command not found: gunicorn`, until added `POETRY_VIRTUALENVS_CREATE=false`; now no longer reproducible
CMD exec poetry run gunicorn -w ${GUN_W} baistro.server:app
