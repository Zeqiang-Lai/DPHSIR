import click

from .common import add_options, common_options

@click.command()
@add_options(common_options)
def denoise(**kwargs):
    pass