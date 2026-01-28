# profile_hotspot_click.py
import cProfile
import pstats
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import P2DingoCV.cli as cli

if __name__ == "__main__":
    sys.argv = [
        "hotspot-cli",
        "run",
        "-a",
        "testImages/oneImage",
        str(Path.home() / "P2DingoCV/output")
    ]

    profiler = cProfile.Profile()
    
    try:
        profiler.enable()
        cli.main()
        profiler.disable()
    except SystemExit:
        # Click calls sys.exit(), catch it so profiler can print
        profiler.disable()

    # Print top 50 functions sorted by cumulative time
    stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumulative")
    stats.print_stats(50)

    # Optional: save for later analysis
    stats.dump_stats("hotspot_profile.prof")

    print("\nProfile finished. Top 50 functions printed above.")
    print("Full profile saved to hotspot_profile.prof")