FROM python:3.12-slim-bookworm AS base

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
CMD poetry lock --no-interaction && poetry sync --no-root --no-interaction && poetry install --no-root --no-interaction && exec poetry run flask --app baistro.server:app --debug run --host 0.0.0.0 --port ${PORT}

FROM base as builder

ENV POETRY_VIRTUALENVS_CREATE=false

# todo: version bumps unnecesarily invalidate the docker cache, as the py code isn't published as a package, it shouldn't be important to track the version there. but `version` is required with package-mode enabled.
COPY ./pyproject.toml /app/pyproject.toml
COPY ./poetry.lock /app/poetry.lock

# todo: using the mount cache increases the image size
#RUN --mount=type=cache,target=/root/.cache/poetry poetry install --sync --no-dev --no-interaction -E gunicorn

# note: `sync` doesn't work without venv (acording to docs "works not well")
#RUN poetry sync --without dev -E gunicorn --no-cache --no-root --no-interaction

RUN poetry install --without dev -E gunicorn --no-cache --no-root --no-interaction
# note: `--no-root` causes the "not installed as a script" warning,
#       which can't be used here when package-mode is not disabled,
#       fix: secondary installtion after copied sources in runtime image itself
# note: with `--compile` the image was ~150MB larger,
#       no measurable performance gain (or unknown how to check)

FROM python:3.12-slim-bookworm AS runtime

ARG VCS_REF
ARG VCS_URL
ARG CI_RUN_URL
ARG BUILD_DATE
ARG VERSION

LABEL org.opencontainers.image.authors="Michael Becker, https://i-am-digital.eu"
LABEL org.opencontainers.image.title="baistro"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.description="Some APIs for AI models that run on CPU, designed for deterministic NLP. Includes Stanza and Sentence Transformers to provide a self-contained NLP processing unit for common tasks."
LABEL org.opencontainers.image.source=$VCS_URL
LABEL org.opencontainers.image.revision=$VCS_REF
LABEL org.opencontainers.image.url=$CI_RUN_URL
LABEL org.opencontainers.image.created=$BUILD_DATE
LABEL org.opencontainers.image.version=$VERSION

ENV PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY ./pyproject.toml /app/pyproject.toml
COPY ./poetry.lock /app/poetry.lock

COPY ./baistro /app/baistro

COPY ./README.md README.md
COPY ./LICENSE LICENSE

# secondary install of only main for correct poetry script install; to fix:
# Warning: 'cli' is an entry point defined in pyproject.toml, but it's not installed as a script. You may get improper `sys.argv[0]`.
RUN poetry install --only main --no-cache --no-interaction

ENV GUN_W 1

# note: kept gettings `command not found: gunicorn`, this was caused by the `/root/.cache` mounts in docker-compose.
#       remove these `volumes` when testing prod image locally!
CMD exec poetry run gunicorn -w ${GUN_W} baistro.server:app
