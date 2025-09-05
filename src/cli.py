import click

@click.group()
def cli():
    pass

@cli.command
#minimal
def minimal():
    pass

#verbose
@cli.command
def verbose():
    pass

#visual
@cli.command
def visual():
    pass

#verbose visual
@cli.command
def allVisuals():