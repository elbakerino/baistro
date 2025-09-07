from apiflask import APIFlask
from baistro._boot import Services


def ai_api(app: APIFlask, s: Services):
    # importing api endpoints lazily, as first the `config.yaml` must be read,
    # before the api docs are defined, while the config is read in models.py,
    # the imports in server.py could be ordered wrong,
    # with the lazy import it is ensured to be setup before api routes are initialized
    from baistro.api.api_clustering import api_clustering
    from baistro.api.api_models import api_models
    from baistro.api.api_word_occurrences import api_word_occurrences
    from baistro.api.api_stanza import api_stanza
    from baistro.api.api_vectors import api_vectors

    api_models(app, s)
    api_vectors(app, s)
    api_stanza(app, s)
    api_word_occurrences(app, s)
    api_clustering(app, s)
