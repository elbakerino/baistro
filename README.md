# baistro · AI APIs

[![Github actions Build](https://github.com/elbakerino/baistro/actions/workflows/blank.yml/badge.svg)](https://github.com/elbakerino/baistro/actions)

Some APIs around AI models.

- CPU only docker setup
- Expects preloaded models, no (annoying) auto-downloads
- Stats about token usages (partially/WIP)
- Automatic OpenAPI generation

## Tasks & Models

### Vector Space Representations

> by [Sentence-Transformers](https://www.sbert.net/)

- Text / Sentences
- Image
- Code

### Linguistic Analysis

> by [Stanza](https://stanfordnlp.github.io/stanza/pipeline.html)

- Locale Identification
- Sentence Segmentation
- Token Classification (NER, POS, MWT)
- Sequence Classification (Sentiment)
- Lemmatization

### Task Implementations

- Semantic Search
- *WIP* Sentence Clustering

## Usage

Clone this repository or add the service to your `docker-compose.yml`:

```yaml
services:
    baistro:
        image: ghcr.io/elbakerino/baistro:0.2.3
        stop_signal: SIGINT
        environment:
            PORT: 8702
            GUN_W: 1
        volumes:
            # folder for model files
            - ./model-assets:/app/model-assets
        ports:
            - "8702:8702"
```

Startup server:

```shell
docker compose up
```

- Service Home: [localhost:8702](http://localhost:8702)
- OpenAPI Docs: [localhost:8702/docs](http://localhost:8702/docs)
- OpenAPI File: [localhost:8702/openapi.json](http://localhost:8702/openapi.json)

Run CLI in docker container:

```shell
# build container before using cli (if never `up`ed before)
docker compose build baistro

# open shell in container:
docker compose run --rm baistro bash

# run cli help:
poetry run cli

# download models:
poetry run cli download

# download model `stanza-multilingual` directly:
poetry run cli download stanza-multilingual

# list models:
poetry run cli models
```

## DEV Notes

Manage dependencies with [poetry v2](https://python-poetry.org/):

```shell
# first update lock file if .toml was changed
poetry lock --regenerate
# then sync and install
poetry sync
poetry install

# poetry lock --regenerate && poetry sync && poetry install
```

## License

This project is distributed as **free software** under the **MIT License**, see [License](https://github.com/elbakerino/baistro/blob/main/LICENSE).

© 2024 Michael Becker https://i-am-digital.eu
