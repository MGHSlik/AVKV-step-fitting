# AVKV: Allan Variance-based Kalafut-Visscher Step Detection

**A Python implementation of the AVKV method.**

## Overview

**AVKV** is  open-source Python code designed to analyze stepping data from optical tweezers and other single-molecule experiments. It addresses the limitations of existing LabVIEW and C implementations by providing a transparent, accessible, and automated workflow.

The algorithm combines **Allan Variance (AV)**, to determine optimal noise filtering and bandwidth, with the **Kalafut-Visscher (KV)** algorithm for statistically robust step fitting. This removes user bias in selecting downsampling rates and standardizes resolution determination across different experimental setups.

## Key Features

*   **Automated Bandwidth Selection:** Uses Allan Variance to objectively calculate the optimal sampling frequency, balancing temporal resolution against thermal noise.
*   **Adaptive Downsampling:** Experimental data can be downsampled to the calculated optimal averaging window from the calcualted bandwidth.
*   **Robust Step Fitting:** Implements the Kalafut-Visscher algorithm to detect steps and dwells without requiring prior knowledge of the number of states.
*   **Analysis & Visualisation Tools** Includes a script which pools multiple fit results and analyses the pooled results, building multiple plots.
*   **Benchmarking Tools:** Includes scripts to compare AVKV performance against standard KV methods (AVKV_vs_KV_comparison.py).
*   **Pure Python:** No dependency on LabVIEW or proprietary commercial licenses.

## Installation

### Prerequisites
*   Python 3.8 or higher
*   `numpy`
*   `scipy`
*   `matplotlib`
*   `typing`
*   `pathlib`
*   `allantools`

### Setup
Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/AVKV-step-detection.git
cd AVKV-step-detection
pip install -r requirements.txt
```

## Methodology

### The Problem
Classic step detection often requires the user to manually filter data or guess the "correct" downsampling frequency. Over-filtering removes real events; under-filtering leaves too much noise for the algorithm to function.

### The AVKV Solution
1.  **Allan Variance Analysis:** The algorithm calculates the Allan Variance of the signal to identify the timeframe where thermal noise averages out, but drift has not yet set in. The minimum of the Allan Variance plot defines the **optimal measurement time ($\tau_{opt}$)**.
2.  **Objective Downsampling:** The data is resampled to $f_{opt} = 1 / \tau_{opt}$.
3.  **Step Fitting:** The Kalafut-Visscher algorithm (based on Schwarz Information Criterion) is applied to this objectively filtered data to identify discrete transitions.

## Citation & References

If you use this software in your research, please cite the associated Master's Thesis:

> **Maarten Slik** (2025). *Force-Dependent Substrate Translocation in FtsH: A Single-Molecule Optical Tweezers Study of a Thermophilic AAA+ Protease* Master's Thesis, Delft University of Technology.

### Original Algorithms
This work is an adaptation and modernization of the following methods:
*   **Original KV Method:** Kalafut, B., & Visscher, K. (2008). *An objective, model-independent method for detection of non-uniform steps in noisy signals.* Computer Physics Communications.
*   **Allan Variance in OT:** Czerwinski, F., et al. (2009). *Quantifying noise in optical tweezers by Allan variance.* Optics Express.

## Acknowledgements

*   **Koen Visscher (University of Arizona):** For providing the original LabVIEW and C source code for the Kalafut-Visscher algorithm, which served as the foundation for this Python implementation.
*   **Marie-Eve Aubin-Tam Lab (TU Delft):** For the experimental data and support during the development of this tool.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
