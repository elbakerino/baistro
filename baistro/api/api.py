from apiflask import APIFlask

from baistro._boot import Services
from baistro.api.api_clustering import api_clustering
from baistro.api.api_models import api_models
from baistro.api.api_sentences import api_sentences
from baistro.api.api_stanza import api_stanza


def ai_api(app: APIFlask, s: Services):
    api_models(app, s)
    api_stanza(app, s)
    api_sentences(app, s)
    api_clustering(app, s)
