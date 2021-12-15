import click

from commands.denoise import denoise

@click.group()
@click.option('--count', default=1, help='number of greetings')
@click.argument('name')
def main(**kwargs):
    pass


if __name__ == '__main__':
    main.add_command(denoise)
    main()