# EEG Sport Imagery Analysis

This repository contains code for processing EEG data collected during a mental imagery task performed by an Olympic sailor. The task involves imagery in different sports contexts (training vs. competition) and with different instructional modalities (guided vs. self-produced). The code includes various steps for processing EEG data, including Power Spectral Density (PSD) analysis, band power computation, and topomap visualization.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Data Format](#data-format)
- [Results](#results)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/aryakoureshi/EEG-Sport-Imagery-Analysis.git
    cd EEG-Sport-Imagery-Analysis
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Place your EEG data files in the `data/` directory. The code expects EEG data in the EEGLAB `.set` format.
   
2. Modify the `path_file`, `sub_name`, and `task_name` variables in the code to reflect the file paths for your subject's data.

3. Run the Python script to process the data:
    ```bash
    python analyze_eeg_data.py
    ```

The script performs the following tasks:
- Extracts data segments based on event markers.
- Computes and plots Power Spectral Density (PSD) for rest, task, and baseline conditions.
- Computes band powers for predefined frequency bands (e.g., alpha, beta, SMR).
- Visualizes band powers on topomaps for each trial.
- Saves the computed band powers to an Excel file.

## Code Structure

- **`analyze_eeg_data.py`**: Main script for processing EEG data. It performs data extraction, PSD computation, band power calculation, and generates topomaps.
- **`data/`**: Directory where EEG `.set` files should be stored.
- **`output/`**: Directory where results (plots, band powers) will be saved.
- **`topomaps/`**: Directory where topomap images will be saved.

## Data Format

The code expects EEG data in the EEGLAB `.set` format with the following event markers:
- '1' for the start of rest condition.
- '2' for the end of rest condition.
- '3' for the start of task condition.
- '4' for the end of task condition.
- '5' for the start of baseline condition.
- '6' for the end of baseline condition.

## Results

The processed results will be saved in the `output/` folder, including:
- PSD plots for each condition (rest, task, baseline).
- Topomap plots for each frequency band (e.g., delta, theta, alpha).
- An Excel file containing band power values for each trial and channel.

## License

This code is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
