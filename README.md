# baistro · AI APIs

[![Github actions Build](https://github.com/elbakerino/baistro/actions/workflows/blank.yml/badge.svg)](https://github.com/elbakerino/baistro/actions)

Some APIs around AI models.

- CPU only docker setup
- Expects preloaded models, no (annoying) auto-downloads
- Stats about token usages (partially/WIP)

## Tasks & Models

> 💎 = most stable
>
> ⚗️ = very experimental / unstable

### Vector Space Representations

> by [Sentence-Transformers](https://www.sbert.net/)

- Text / Sentences 💎
- Image
- Code 💎

### Linguistic Analysis

> by [Stanza](https://stanfordnlp.github.io/stanza/pipeline.html) 💎

- Locale Identification
- Sentence Segmentation
- Token Classification (NER, POS, MWT)
- Sequence Classification (Sentiment)
- Lemmatization

### Document Processing

- Image to Data (by `donut`) ⚗️
- Visual Document Question Answering (Image) (by `donut`)
- *WIP* Document Classification (Image) (by `dit`) ⚗️
    - (dataset) RVL-CDIP: `"letter", "form", "email", "handwritten", "advertisement", "scientific report", "scientific publication", "specification", "file folder", "news article", "budget", "invoice", "presentation", "questionnaire", "resume", "memo"`

### NLI / QA / QAG / QG

> general Natural Language Inference

- Question Answering
- Question Answer Generation ⚗️
- Question Generation ⚗️
- Question Natural Language Inference / QNLI 💎

### Task Implementations

- Semantic Search 💎
- *WIP* Sentence Clustering
- *todo* Topic Clustering (by [BERTopic](https://maartengr.github.io/BERTopic/))

## Usage

Startup server:

```shell
docker compose up
```

- Service Home: [localhost:8702](http://localhost:8702)
- OpenAPI Docs: [localhost:8702/docs](http://localhost:8702/docs) (WIP 🚧)

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

Manage dependencies with poetry:

```shell
poetry lock --no-update
poetry install --sync
# poetry lock --no-update && poetry install --sync
```

## License

This project is distributed as **free software** under the **MIT License**, see [License](https://github.com/elbakerino/baistro/blob/main/LICENSE).

© 2024 Michael Becker https://i-am-digital.eu
