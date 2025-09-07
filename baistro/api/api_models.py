import os
from apiflask import APIFlask
from baistro.api.schemas import ModelsResponse
from baistro._boot import Services
from baistro.model_control.model_base import model_url
from baistro.model_control.models import models


def get_folder_size(folder_path):
    total_size = 0
    for path, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(path, file)
            total_size += os.path.getsize(file_path)
    return total_size


def api_models(app: APIFlask, s: Services):
    @app.route('/models', methods=['GET'])
    @app.output(ModelsResponse)
    @app.doc(tags=['Models'], description='''
Returns a list of all available models and their statistics.
''')
    def model_list():
        models_stats = {}

        for model in models.list():
            models_stats[model.id] = {
                'tasks': model.tasks,
                'name': model.name,
                'locale': model.locale if hasattr(model, 'locale') else None,
                'url': model_url(model),
                'size': get_folder_size(model.folder),
            }
        return {
            'total': len(models_stats),
            'stats': models_stats,
        }
