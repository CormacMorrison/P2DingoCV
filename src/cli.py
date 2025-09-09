import click
from HotspotLogic.HotSpotDetectorSubclasses.MaximumDetector import *
from HotspotLogic.HotSpotDetectorSubclasses.MinimalDetector import *
from HotspotLogic.HotSpotDetectorSubclasses.VerboseDetector import *
from HotspotLogic.HotSpotDetectorSubclasses.VisualDetector import *
from Camera.Camera import Camera
from Camera.CameraFactory import CameraFactory

@click.group()
def cli():
    """Main CLI entry point."""
    pass

# --- Sub-function implementations ---
def runMinimal(camera: Camera, outputPath: str):
    click.echo("Running in minimal mode")
    hotspotDetector: HotspotDetector = MinimalDetector(camera, outputPath)
    hotspotDetector.execute()

def runVerbose(camera: Camera, outputPath: str):
    click.echo("Running in verbose mode")
    hotspotDetector: HotspotDetector = VerboseDetector(camera, outputPath)
    hotspotDetector.execute()
    
def showVisual(camera: Camera, outputPath: str):
    click.echo("Showing visuals")
    hotspotDetector: HotspotDetector = VisualDetector(camera, outputPath)
    hotspotDetector.execute()
 
def runAll(camera: Camera, outputPath: str):
    click.echo("Running with verbose output and visuals")
    hotspotDetector: HotspotDetector = MaximumDetector(camera, outputPath)
    hotspotDetector.execute()
 
# --- Dispatcher command with flags ---
@cli.command()
@click.option("-m", "--minimal", is_flag=True, help="Run with only JSON output")
@click.option("-v", "--verbose", is_flag=True, help="Run with all component data")
@click.option("-vi", "--visual", is_flag=True, help="Show visual output")
@click.option("-a", "--all", "all_modes", is_flag=True, help="Run with verbose + visuals")
@click.argument('inputPath')
@click.argument('outputPath')
@click.pass_context
def run(minimal, verbose, visual, all_modes, inputPath: str, outputPath: str):
    """Run the program with different modes."""
    camera: Camera = CameraFactory.create(f"{inputPath}")
    if not any([minimal, verbose, visual, all_modes]):
        runMinimal(camera, outputPath)
        return
    if minimal:
        runMinimal(camera, outputPath)
    if verbose:
        runVerbose(camera, outputPath)
    if visual:
        showVisual(camera, outputPath)
    if all_modes:
        runAll(camera, outputPath)

if __name__ == "__main__":
    cli()