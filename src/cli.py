import click

@click.group()
def cli():
    pass

@cli.command
def default():
    pass

@cli.command
def verbose():
    pass

@cli.command
def visual():
    pass

@cli.command
def allVisuals():