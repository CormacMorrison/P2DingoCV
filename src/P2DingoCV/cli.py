import click
from P2DingoCV.HotspotLogic.HotspotDetector import HotspotDetector
from P2DingoCV.HotspotLogic.HotSpotDetectorSubclasses.MaximumDetector import *
from P2DingoCV.HotspotLogic.HotSpotDetectorSubclasses.MinimalDetector import *
from P2DingoCV.HotspotLogic.HotSpotDetectorSubclasses.VerboseDetector import *
from P2DingoCV.HotspotLogic.HotSpotDetectorSubclasses.VisualDetector import *
from P2DingoCV.Camera.Camera import Camera
from P2DingoCV.Camera.CameraFactory import CameraFactory

@click.group()
def cli():
    """Main CLI entry point for hotspot detection.

    This group allows running different detection modes via subcommands.
    """
    pass

# --- Sub-function implementations ---
def runMinimal(camera: Camera, outputPath: str, config: str | None):
    """Run hotspot detection in minimal mode.

    Only outputs JSON results to the specified output path.

    Args:
        camera (Camera): The camera object used to capture frames or video.
        outputPath (str): Directory where JSON results will be saved.
        config (str | None): Optional path to JSON configuration file with detection parameters.
    """
    click.echo("Running in minimal mode")
    hotspotDetector: HotspotDetector = MinimalDetector(camera, outputPath, config)
    hotspotDetector.execute()

def runVerbose(camera: Camera, outputPath: str, config: str | None):
    """Run hotspot detection in verbose mode.

    Outputs all component data, including intermediate metrics, in addition to JSON results.

    Args:
        camera (Camera): The camera object used to capture frames or video.
        outputPath (str): Directory where results and logs will be saved.
        config (str | None): Optional path to JSON configuration file with detection parameters.
    """
    click.echo("Running in verbose mode")
    hotspotDetector: HotspotDetector = VerboseDetector(camera, outputPath, config)
    hotspotDetector.execute()
    
def showVisual(camera: Camera, outputPath: str, config: str | None):
    """Run hotspot detection and display visual outputs.

    Shows the processed frames with detected hotspots highlighted.

    Args:
        camera (Camera): The camera object used to capture frames or video.
        outputPath (str): Directory where any results will be saved.
        config (str | None): Optional path to JSON configuration file with detection parameters.
    """
    click.echo("Showing visuals")
    hotspotDetector: HotspotDetector = VisualDetector(camera, outputPath, config)
    hotspotDetector.execute()
 
def runAll(camera: Camera, outputPath: str, config: str | None):
    """Run hotspot detection in maximum mode.

    Outputs verbose data and displays visuals simultaneously.

    Args:
        camera (Camera): The camera object used to capture frames or video.
        outputPath (str): Directory where results will be saved.
        config (str | None): Optional path to JSON configuration file with detection parameters.
    """
    click.echo("Running with verbose output and visuals")
    hotspotDetector: HotspotDetector = MaximumDetector(camera, outputPath, config)
    hotspotDetector.execute()
 
# --- Dispatcher command with flags ---
HELP_TEXT = """Run hotspot detection in different modes.

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
        showVisual(camera, outputpath, config)
    if all_modes:
        runAll(camera, outputpath, config)
        
def main() -> None:
    cli()
    
if __name__ == "__main__":
    main()

