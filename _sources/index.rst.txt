.. P2DingoCV documentation master file, created by
   sphinx-quickstart on Wed Sep 10 01:22:50 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to P2DingoCV's documentation!
==========================================

P2DingoCV is a computer vision program designed for the P2Dingo project. The P2Dingo project is an autonomus solar panel inspection quadraped. 
This program is designed to take images the quadraped captures and determine if a solar hotspot is present in a given image. 
This documentation provides comprehensive coverage of all modules and their functionality.

This project due to a lack quality data required to develop a neural network in the specific orientation P2Dingo captures images in relies on a
traditional computer vision approach and k means segmentation algoritim in combination with heuristics to determine if a panel has a hotspot.

The default paramaters were carefully hand tuned by me but I have left open a method via a JSON file to update them as you see fit to tune for 
the specifc conditions of P2Dingo which I have not yet been able to see.

 

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Overview
--------

P2DingoCV consists of two main modules:

* **Camera Module**: Handles camera operations and image capture functionality
* **HotspotLogic Module**: Implements hotspot detection and analysis logic

CLI Usage Guide
===============

This guide explains how to install, set up, and run the P2DingoCV hotspot detection CLI tool.

Installation and Setup
----------------------

Prerequisites
~~~~~~~~~~~~~

- Python 3.11 or higher
- Poetry (for dependency management)

Quick Setup
~~~~~~~~~~~

1. **Clone the repository and run the setup script:**

.. code-block:: bash

   git clone git@github.com:CormacMorrison/P2DingoCV.git
   cd P2DingoCV
   ./setup.sh

The setup script will automatically:

- Install Poetry (if not already installed)
- Create a virtual environment
- Install all dependencies
- Set up the project for development

2. **Run the Program:**

.. code-block:: bash

   poetry run hotspot-cli

Manual Setup (Alternative)
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you prefer to set up manually:

.. code-block:: bash

   # Install Poetry
   pipx install poetry

   # Install dependencies
   poetry install

   # Run the commands
   poetry run hotspot-cli

Compile to Binary using Nuitka (Optional After poetry install)
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you would prefer to have a binary install a compile script using 
Nuitka has been provided.

.. code-block:: bash

   ./compile.sh
   # Run if you want the executable to be globally avaliable
   sudo mv build/hotspot-cli /usr/local/bin/

It will output an single executable to a folder called build which you 
can relocate to a different location. 

If you wish to make it globally avaliable run 

.. code-block:: bash
   # Run if you want the executable to be globally avaliable
   sudo mv build/hotspot-cli /usr/local/bin/


If you choose to do this replace every instance of 

.. code-block:: bash

   poetry run hotspot-cli

with 

.. code-block:: bash

   ./path/to/executable/hotspot-cli 

or just

.. code-block:: bash

   hotspot-cli

if you have made it globally avaliable

CLI Overview
------------

The P2DingoCV CLI provides hotspot detection with multiple output modes:

- **Minimal Mode** (``-m``): JSON results only
- **Verbose Mode** (``-v``): All component data and metrics
- **Visual Mode** (``-vi``): Display processed frames with hotspots highlighted
- **All Mode** (``-a``): Verbose output + visual display

Basic Usage
-----------

Command Structure
~~~~~~~~~~~~~~~~

.. code-block:: bash

   poetry run hotspot-cli run [OPTIONS] INPUT_PATH OUTPUT_PATH 

   # or if you have compiled and made globally avaliable

   hotspot-cli run [OPTIONS] INPUT_PATH OUTPUT_PATH 

Arguments
~~~~~~~~~

- ``INPUT_PATH``: Path to input source (camera config file, video file, or image)
- ``OUTPUT_PATH``: Directory where results will be saved



Options
~~~~~~~

- ``-m, --minimal``: Run with JSON output only
- ``-v, --verbose``: Run with detailed component data
- ``-vi, --visual``: Show visual output with detected hotspots
- ``-a, --all``: Run with both verbose output and visuals
- ``-c, --config PATH``: Optional JSON configuration file

Usage Examples
--------------

Basic Hotspot Detection (Minimal Mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Default minimal mode (no flags needed)
   poetry run hotspot-cli run input/folderOfFrames output/results/

   # Explicit minimal mode
   poetry run hotspot-cli run -m input/folderOfFrames output/results/

**Output:** JSON files with detection results in the output directory.

Verbose Mode with Detailed Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   poetry run hotspot-cli run -v input/folderOfFrames output/detailed_results/

**Output:** JSON results plus comprehensive logs and intermediate metrics.

Visual Mode for Development/Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   poetry run hotspot-cli run -vi input/FolderOfFrames output/visual_test/

**Output:** Real-time visual display showing detected hotspots overlaid on frames.

Maximum Mode (Verbose + Visual)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   poetry run hotspot-cli run -a input/FolderOfFrames output/full_analysis/

**Output:** Complete analysis with visual display and all component data.

Using Custom Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   poetry run python -m P2DingoCV.cli run -v -c config/custom_detection.json input/data/ output/results/

Configuration Files
-------------------

Detection Parameters Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a JSON configuration file to customize detection parameters:

.. code-block:: json

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

Parameter Descriptions
~~~~~~~~~~~~~~~~~~~~~

- ``k``: Number of clusters for K-means clustering
- ``clusterJoinKernel``: Kernel size for joining nearby clusters
- ``hotSpotThreshold``: Threshold for hotspot classification (0.0-1.0)
- ``sigmoidSteepnessDeltaP``: Steepness parameter for delta-P sigmoid function
- ``sigmoidSteepnessZ``: Steepness parameter for Z-score sigmoid function
- ``compactnessCutoff``: Minimum compactness score for valid hotspots
- ``dilationSize``: Size of dilation kernel for morphological operations
- ``wDeltaP``: Weight for delta-P in composite scoring
- ``wZscore``: Weight for Z-score in composite scoring
- ``wCompactness``: Weight for compactness in composite scoring
- ``wAspectRatio``: Weight for aspect ratio in composite scoring
- ``wEccentricity``: Weight for eccentricity in composite scoring

Input Source Types
------------------
Directory of Images
~~~~~~~~~~~~~~

.. code-block:: bash

   poetry run hotspot-cli run input/image_sequence/ output/batch_results/

Video Files
~~~~~~~~~~~

.. code-block:: bash

   poetry hotspot-cli run input/thermal_video.mp4 output/results/

Supported formats: MP4, AVI, MOV, etc.

Output Structure
----------------

Minimal Mode Output
~~~~~~~~~~~~~~~~~~

Only if a hotspot is detected:

.. code-block:: text

   output/resultsFolder/
   ├── hotspotOutput.json

Maximum Mode Output
~~~~~~~~~~~~~~~~~~

Everything outputted:

.. code-block:: text

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

Verbose Mode Output
~~~~~~~~~~~~~~~~~~

Intermediate text diagnostic data:

.. code-block:: text

   output/resultsFolder/
   ├── hotspotOutput.json

Visual Mode Output
~~~~~~~~~~~~~~~~~

All visual data + basic text data:

.. code-block:: text

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

Advanced Usage
--------------

Combining Multiple Modes
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run verbose AND visual modes simultaneously
   poetry run hotspot-cli run -v -vi input/data.mp4 output/combined/


Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~

1. **Import Errors**

   .. code-block:: bash

      # Ensure you're in the Poetry environment
      poetry run python src/P2DingoCV/cli.py
      
      # Verify installation
      poetry install --no-dev

2. **Camera Access Issues**

   .. code-block:: bash

      # Check camera permissions (Linux/macOS)
      ls -la /dev/video*
      
      # Test camera access
      python3 -c "import cv2; print(cv2.VideoCapture(0).isOpened())"

3. **Output Directory Permissions**

   .. code-block:: bash

      # Create output directory with proper permissions
      mkdir -p output/results
      chmod 755 output/results

4. **Configuration File Errors**

   .. code-block:: bash

      # Validate JSON configuration
      python3 -m json.tool config/detection_config.json


Getting Help
------------

CLI Help
~~~~~~~~

.. code-block:: bash

   # General help
   poetry run hotspot-cli --help

   # Command-specific help
   poetry run hotspot-cli run --help

Version Information
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   poetry run hotspot-cli --version


API Documentation
=================

.. autosummary::
   :toctree: _autosummary
   :recursive:

   P2DingoCV

Modules
=======

Camera Module
-------------

.. automodule:: P2DingoCV.Camera
   :members:
   :undoc-members:
   :show-inheritance:

HotspotLogic Module
-------------------

.. automodule:: P2DingoCV.HotspotLogic
   :members:
   :undoc-members:
   :show-inheritance:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
=======================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


