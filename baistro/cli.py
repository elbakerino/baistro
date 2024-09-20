from baistro._boot import boot
import logging

import sys
import click

from baistro.commands.cli_model import cli_model

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

s = boot()


@click.group()
@click.pass_context
def cli(ctx):
    # ctx.obj = db = open_db(repo_home)

    @ctx.call_on_close
    def close_db():
        logging.info(f'shutdown')
        s.shutdown()

cli_model(cli, s)
