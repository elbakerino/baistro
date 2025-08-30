import os

TRUTHY_ENV_VAL = ['true', 'True', '1', 'yes']


class AppConfig:
    APP_ENV = os.getenv('APP_ENV')
    SERVICE_NAME = os.getenv('SERVICE_NAME', "unnamed-service")
    SERVICE_NAMESPACE = os.getenv('SERVICE_NAMESPACE', "unknown")

    LOGGING_LEVEL = os.getenv('LOGGING_LEVEL', "INFO")

    API_DOCS_UI = os.getenv('API_DOCS_UI', 'swagger-ui')
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')
    CORS_SEND_WILDCARD = os.getenv('CORS_SEND_WILDCARD') in TRUTHY_ENV_VAL

    MODEL_DIR = os.getenv('MODEL_DIR', '/app/model-assets')
    SHARED_DIR = os.getenv('SHARED_DIR', '/app/shared-assets')
    MODELS = {key[len('MODEL__'):]: os.environ[key] for key in os.environ.keys() if key.startswith('MODEL__')}
    PRELOAD_VECTOR_TEXT = os.getenv('PRELOAD_VECTOR_TEXT') in TRUTHY_ENV_VAL
    PRELOAD_VECTOR_CODE = os.getenv('PRELOAD_VECTOR_CODE') in TRUTHY_ENV_VAL
    PRELOAD_VECTOR_IMAGE = os.getenv('PRELOAD_VECTOR_IMAGE') in TRUTHY_ENV_VAL


# RUNTIME-env vars enable bridging to defaults of package,
# thus they must be imported before that package,
# which makes sense here.

# todo: optimize these runtime-env vars
# todo: it seems the cache-dirs are also configurable, but with other settings for some
os.environ['STANZA_RESOURCES_DIR'] = AppConfig.MODEL_DIR + '/_stanza'

# todo: the cache dir for sentence-transformers is in conflict with configured `folder`,
#       between 2.2.2 and 2.3.1 the internal folder normalization seems to have changed,
#       which broke the configured `folder` property in SentenceTransformer model classes,
#       the fix `.save(Model.folder)` resulted in duplicates on disk as the cache-folder is the same as the configured model folder,
#       thus removed the env var `SENTENCE_TRANSFORMERS_HOME` for the moment,
#       to use the default folder in the container which downloads the models,
#       and save the model into the mounted folder afterwards.
# os.environ['SENTENCE_TRANSFORMERS_HOME'] = AppConfig.MODEL_DIR + '/_sent_tran'
SENTENCE_TRANSFORMERS_HOME = AppConfig.MODEL_DIR + '/_sent_tran'
