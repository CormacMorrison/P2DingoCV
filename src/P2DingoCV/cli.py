import click
from P2DingoCV.HotspotLogic.HotspotDetector import HotspotDetector
from P2DingoCV.App.App import App
from P2DingoCV.App.Maximal import Maximal
from P2DingoCV.App.Minimal import Minimal
from P2DingoCV.App.Verbose import Verbose
from P2DingoCV.App.Visual import Visual
from P2DingoCV.Camera.Camera import Camera
from P2DingoCV.Camera.CameraFactory import CameraFactory

@click.group()
def cli():
    """Main CLI entry point for hotspot detection.

    This group allows running different detection modes via subcommands.
    """
    pass

# --- Sub-function implementations ---
def runMinimal(camera: Camera, outputPath: str, config: str | None) -> None:
    """Run hotspot detection and panel segmentaion in minimal mode.

    Only outputs JSON results to the specified output path.

    Args:
        camera (Camera): The camera object used to capture frames or video.
        outputPath (str): Directory where JSON results will be saved.
        config (str | None): Optional path to JSON configuration file with detection parameters.
    """
    click.echo("Running in minimal mode")
    app: App = Minimal(camera, outputPath, config)
    app.execute()

def runVerbose(camera: Camera, outputPath: str, config: str | None) -> None:
    """Run hotspot detection and panel segmentaion in verbose mode.

    Outputs all component data, including intermediate metrics, in addition to JSON results.

    Args:
        camera (Camera): The camera object used to capture frames or video.
        outputPath (str): Directory where results and logs will be saved.
        config (str | None): Optional path to JSON configuration file with detection parameters.
    """
    click.echo("Running in verbose mode")
    app: App = Verbose(camera, outputPath, config)
    app.execute()

    
def runVisual(camera: Camera, outputPath: str, config: str | None) -> None:
    """Run hotspot detection and panel segmentaion and display visual outputs.

    Shows the processed frames with detected hotspots highlighted.

    Args:
        camera (Camera): The camera object used to capture frames or video.
        outputPath (str): Directory where any results will be saved.
        config (str | None): Optional path to JSON configuration file with detection parameters.
    """
    click.echo("Rinning with visuals")
    app: App = Visual(camera, outputPath, config)
    app.execute()
 
def runMaximal(camera: Camera, outputPath: str, config: str | None) -> None:
    """Run hotspot detection and panel segmentaion in maximum mode.

    Outputs verbose data and displays visuals simultaneously.

    Args:
        camera (Camera): The camera object used to capture frames or video.
        outputPath (str): Directory where results will be saved.
        config (str | None): Optional path to JSON configuration file with detection parameters.
    """
    click.echo("Running with verbose output and visuals")
    app: App = Maximal(camera, outputPath, config)
    app.execute()
 

# --- Dispatcher command with flags ---
HELP_TEXT = """Run hotspot detection and panel segmentaion in different modes.

\b
Arguments:
  inputPath   Path to input source (camera config, video, etc.)
  outputPath  Path to save JSON results

\b
Example configuration file format:
  {
      "k": 10,
      "clusterJoinKernel": 3,
      "hotSpotThreshold": 0.7,
      "sigmoidSteepnessDeltaP": 0.25,
      "sigmoidSteepnessZ": 0.23,
      "compactnessCutoff": 0.6,
      "dilationSize": 5,
      "wDeltaP": 0.3,
      "wZscore": 0.3,
      "wCompactness": 0.4,
      "wAspectRatio": 0.0,
      "wEccentricity": 0.0
  }
"""
@cli.command(help=HELP_TEXT)
@click.option("-m", "--minimal", is_flag=True, help="Run with only JSON output")
@click.option("-v", "--verbose", is_flag=True, help="Run with all component data")
@click.option("-vi", "--visual", is_flag=True, help="Show visual output")
@click.option("-a", "--all", "all_modes", is_flag=True, help="Run with verbose + visuals")
@click.option(
    "-c", "--config", 
    type=click.Path(exists=True, dir_okay=False), 
    help="Optional JSON config file (see format above)"
)
@click.argument('inputPath')
@click.argument('outputPath')
def run(minimal, verbose, visual, all_modes, config: str, inputpath: str, outputpath: str):
    """Run the program with different modes."""
    # Load config if provided
    # Create camera
    camera: Camera = CameraFactory.create(inputpath)

    # If no flags are provided, default to minimal
    if not any([minimal, verbose, visual, all_modes]):
        runMinimal(camera, outputpath, config)
        return

    # Run based on flags
    if minimal:
        runMinimal(camera, outputpath, config)
    if verbose:
        runVerbose(camera, outputpath, config)
    if visual:
        runVisual(camera, outputpath, config)
    if all_modes:
        runMaximal(camera, outputpath, config)
        
def main() -> None:
    cli()
    
if __name__ == "__main__":
    main()

