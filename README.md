# P2DingoCV Documentation

Welcome to P2DingoCV's documentation!

P2DingoCV is a computer vision program designed for the P2Dingo project, an autonomous solar panel inspection quadruped. This program processes images captured by the quadruped to determine if a solar hotspot is present in a given image. The solution combines traditional computer vision techniques and K-means segmentation algorithms to detect hotspots.

Due to limited data quality for training a neural network, this approach relies on carefully tuned parameters and heuristics for detecting solar hotspots.

---

## Table of Contents

- [Overview](#overview)
- [Installation and Setup](#installation-and-setup)
  - [Quick Setup](#quick-setup)
  - [Manual Setup](#manual-setup)
  - [Compile to Binary using Nuitka (Optional)](#compile-to-binary-using-nuitka-optional)
- [CLI Usage Guide](#cli-usage-guide)
  - [Basic Usage](#basic-usage)
  - [Usage Examples](#usage-examples)
  - [Advanced Usage](#advanced-usage)
- [Configuration Files](#configuration-files)
  - [Detection Parameters](#detection-parameters-configuration)
  - [Input Source Types](#input-source-types)
  - [Output Structure](#output-structure)
  - [Troubleshooting](#troubleshooting)
- [Getting Help](#getting-help)
- [API Documentation](#api-documentation)

---

## Overview

P2DingoCV consists of two main modules:

- **Camera Module**: Handles camera operations and image capture functionality.
- **HotspotLogic Module**: Implements hotspot detection and analysis logic.

## Installation and Setup

### Prerequisites

- Python 3.11 or higher
- Poetry (for dependency management)

### Quick Setup

1. **Clone the repository and run the setup script:**

    ```bash
    git clone git@github.com:CormacMorrison/P2DingoCV.git
    cd P2DingoCV
    ./setup.sh
    ```

    The setup script will automatically:

    - Install Poetry (if not already installed)
    - Create a virtual environment
    - Install all dependencies
    - Set up the project for development

2. **Run the Program:**

    ```bash
    poetry run hotspot-cli
    ```

### Manual Setup

If you prefer to set up manually:

```bash
# Install Poetry
pipx install poetry

# Install dependencies
poetry install

# Run the commands
poetry run hotspot-cli
````

### Compile to Binary using Nuitka (Optional)

After the Poetry setup, if you'd like to compile the program to a binary executable, use the provided `compile.sh` script:

```bash
./compile.sh
# Move to a global directory
sudo mv build/hotspot-cli /usr/local/bin/
```

You can now run it globally with:

```bash
hotspot-cli
```

---

## CLI Usage Guide

The P2DingoCV CLI provides hotspot detection with multiple output modes:

* **Minimal Mode** (`-m`): JSON results only
* **Verbose Mode** (`-v`): All component data and metrics
* **Visual Mode** (`-vi`): Display processed frames with hotspots highlighted
* **All Mode** (`-a`): Verbose output + visual display

### Basic Usage

#### Command Structure

```bash
poetry run hotspot-cli run [OPTIONS] INPUT_PATH OUTPUT_PATH
# Or if globally available
hotspot-cli run [OPTIONS] INPUT_PATH OUTPUT_PATH
```

#### Arguments

* `INPUT_PATH`: Path to the input source (camera config file, video file, or image)
* `OUTPUT_PATH`: Directory where results will be saved

#### Options

* `-m, --minimal`: Run with JSON output only
* `-v, --verbose`: Run with detailed component data
* `-vi, --visual`: Show visual output with detected hotspots
* `-a, --all`: Run with both verbose output and visuals
* `-c, --config PATH`: Optional JSON configuration file

---

### Usage Examples

#### Basic Hotspot Detection (Minimal Mode)

```bash
# Default minimal mode (no flags needed)
poetry run hotspot-cli run input/folderOfFrames output/results/

# Explicit minimal mode
poetry run hotspot-cli run -m input/folderOfFrames output/results/
```

#### Verbose Mode with Detailed Metrics

```bash
poetry run hotspot-cli run -v input/folderOfFrames output/detailed_results/
```

#### Visual Mode for Development/Testing

```bash
poetry run hotspot-cli run -vi input/FolderOfFrames output/visual_test/
```

#### Maximum Mode (Verbose + Visual)

```bash
poetry run hotspot-cli run -a input/FolderOfFrames output/full_analysis/
```

#### Using Custom Configuration

```bash
poetry run python -m P2DingoCV.cli run -v -c config/custom_detection.json input/data/ output/results/
```

---

## Configuration Files

### Detection Parameters Configuration

Create a JSON configuration file to customize detection parameters:

```json
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
```

### Parameter Descriptions

* `k`: Number of clusters for K-means clustering
* `clusterJoinKernel`: Kernel size for joining nearby clusters
* `hotSpotThreshold`: Threshold for hotspot classification (0.0-1.0)
* `sigmoidSteepnessDeltaP`: Steepness parameter for delta-P sigmoid function
* `sigmoidSteepnessZ`: Steepness parameter for Z-score sigmoid function
* `compactnessCutoff`: Minimum compactness score for valid hotspots
* `dilationSize`: Size of dilation kernel for morphological operations
* `wDeltaP`, `wZscore`, `wCompactness`, `wAspectRatio`, `wEccentricity`: Weights for composite scoring

---

### Input Source Types

#### Directory of Images

```bash
poetry run hotspot-cli run input/image_sequence/ output/batch_results/
```

#### Video Files

```bash
poetry run hotspot-cli run input/thermal_video.mp4 output/results/
```

Supported formats: MP4, AVI, MOV, etc.

---

### Output Structure

#### Minimal Mode Output

If a hotspot is detected:

```
output/resultsFolder/
└── hotspotOutput.json
```

#### Maximum Mode Output

```
output/resultsFolder/
├── hotspotOutput.json
├── frames/
│   ├── frame_0_hotspot.png
│   ├── frame_1_hotspot.png
│   └── frame_3_hotspot.png
├── diagnostics/
│   └── diagnostic1_hotspot.png
└── plots/
    └── plot1.png
```

#### Verbose Mode Output

```
output/resultsFolder/
└── hotspotOutput.json
```

#### Visual Mode Output

```
output/resultsFolder/
├── hotspotOutput.json
├── frames/
│   ├── frame_0_hotspot.png
│   ├── frame_1_hotspot.png
│   └── frame_3_hotspot.png
├── diagnostics/
│   └── diagnostic1_hotspot.png
└── plots/
    └── plot1.png
```

---

## Advanced Usage

#### Combining Multiple Modes

```bash
# Run verbose AND visual modes simultaneously
poetry run hotspot-cli run -v -vi input/data.mp4 output/combined/
```

---

## Troubleshooting

### Common Issues

1. **Import Errors**

   ```bash
   # Ensure you're in the Poetry environment
   poetry run python src/P2DingoCV/cli.py

   # Verify installation
   poetry install --no-dev
   ```

2. **Camera Access Issues**

   ```bash
   # Check camera permissions (Linux/macOS)
   ls -la /dev/video*

   # Test camera access
   python3 -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
   ```

3. **Output Directory Permissions**

   ```bash
   # Create output directory with proper permissions
   mkdir -p output/results
   chmod 755 output/results
   ```

4. **Configuration File Errors**

   ```bash
   # Validate JSON configuration
   python3 -m json.tool config/detection_config.json
   ```

---

## Getting Help

### CLI Help

```bash
# General help
poetry run hotspot-cli --help

# Command-specific help
poetry run hotspot-cli run --help
```

### Version Information

```bash
poetry run hotspot-cli --version
```

---

## API Documentation

See the [API Documentation](https://github.com/CormacMorrison/P2DingoCV) for detailed module descriptions.
