import os
from typing import Optional

import click
from click import Group

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


def cli_model(cli: Group, s: Services):
    @cli.command()
    @click.argument('model_id', required=False)
    def download(model_id: Optional[str] = None):
        if not model_id:
            model_id = input(f"Which model? [{', '.join([model.id for model in models.list()])}] ")

        if not models.has(model_id):
            raise click.UsageError(f'unknown model: {model_id}')

        click.echo(f'starting download for: {model_id}')
        model = models.get_type(model_id)
        model.download()

    @cli.command(name='models')
    def models_command():
        for model in models.list():
            click.echo(f'id: {model.id}')
            click.echo(f'  tasks: {", ".join(model.tasks)}')
            click.echo(f'  name : {model.name} / {model_url(model)}')
            click.echo(f'  size : {get_folder_size(model.folder)}')
