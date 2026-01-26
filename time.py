# profile_hotspot.py
import cProfile
import pstats
import subprocess
import sys


# --- Option 1: Run the CLI as a subprocess ---
def run_cli_subprocess():
    cmd = [
        "poetry",
        "run",
        "hotspot-cli",
        "run",
        "-a",
        "testImages/oneImage",
        "~/P2DingoCV/output/",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)


# --- Option 2: Run the CLI programmatically (if importable) ---
# If hotspot_cli exposes a Python API you can call directly, like `hotspot_cli.main(args)`
# then you can profile it directly:


def run_cli_profiled():
    import hotspot_cli  # replace with the actual import if available
    import sys

    args = ["run", "-a", "testImages/oneImage", "~/P2DingoCV/output/"]
    sys.argv = ["hotspot-cli"] + args
    hotspot_cli.main()  # replace with actual entry function


if __name__ == "__main__":
    # Create profiler
    profiler = cProfile.Profile()
    profiler.enable()

    # Choose one of these:
    run_cli_subprocess()  # runs as subprocess (profiling won't capture internal Python calls inside hotspot-cli)

    profiler.disable()

    # Print stats sorted by cumulative time
    stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumulative")
    stats.print_stats(50)  # show top 50 functions
