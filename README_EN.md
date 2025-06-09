[English](README_EN.md) | [中文](README.md)

# Dynamic Interaction Analysis of Consumer Confidence, Income, Credit Conditions, and Real Consumption Expenditure

## Project Overview

This project uses a multivariate time series model (VAR - Vector Autoregression) to analyze the dynamic interactions among consumer confidence, income, credit conditions, and real consumption expenditure.

## Data Description

- **UMCSENT**: Consumer Sentiment Index (University of Michigan Consumer Sentiment)
- **DSPIC96**: Real Disposable Personal Income (2012 USD)
- **PCE**: Personal Consumption Expenditures (billion USD)
- **TOTALSL**: Total Consumer Credit (million USD)

## Environment Setup

### 1. Create a conda virtual environment

```bash
# Create a new conda environment
conda create -n dynamic_interaction python=3.9

# Activate the environment
conda activate dynamic_interaction
```

### 2. Install dependencies

```bash
# Install required Python packages
pip install -r requirements.txt

# Or install via conda
conda install pandas numpy matplotlib seaborn scipy
conda install -c conda-forge statsmodels
```

### 3. Verify installation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
print("All packages installed successfully!")
```

## File Structure

```
Dynamic Interaction/
├── UMCSENT.csv          # Consumer sentiment data
├── DSPIC96.csv          # Real income data  
├── PCE.csv              # Consumption expenditure data
├── TOTALSL.csv          # Total credit data
├── requirements.txt     # Dependency list
├── dynamic_interaction_analysis.py  # Full analysis script
├── quick_analysis.py    # Quick analysis script
└── README.md            # Documentation
```

## How to Run

### Method 1: Full Analysis (Recommended)

```bash
python dynamic_interaction_analysis.py
```

This will run the complete analysis process, including:
- Data loading and preprocessing
- Descriptive statistical analysis
- Stationarity test (ADF test)
- Cointegration test (Johansen test)
- Granger causality test
- VAR model estimation
- Impulse response analysis
- Variance decomposition analysis
- Generate analysis report

### Method 2: Quick Analysis

```bash
python quick_analysis.py
```

This will run a simplified analysis, suitable for quick validation and preliminary exploration.

## Analysis Methods

### 1. Stationarity Test
Use the ADF test to determine the stationarity of time series and difference non-stationary series.

### 2. Cointegration Test
Use the Johansen test to determine whether there is a long-term equilibrium relationship among variables.

### 3. Granger Causality Test
Test the direction and strength of causal relationships among variables.

### 4. VAR Model
Build a vector autoregression model to analyze the dynamic interactions among variables.

### 5. Impulse Response Analysis
Analyze the dynamic impact of a shock to one variable on other variables.

### 6. Variance Decomposition
Analyze the contribution of each variable to the forecast error variance.

## Output Results

After running the full analysis, the following files will be generated:

1. **时间序列图.png** - Time series trend chart for each variable
2. **相关性热力图.png** - Correlation matrix heatmap
3. **脉冲响应图.png** - Impulse response function chart
4. **正交化脉冲响应图.png** - Orthogonalized impulse response function chart
5. **方差分解图.png** - Variance decomposition chart
6. **分析报告.txt** - Detailed analysis report

## Notes

1. Ensure all CSV files are in the same directory
2. The date format of data files should be 'YYYY-MM-DD'
3. If you encounter Chinese font display issues, please install the corresponding Chinese fonts
4. It is recommended to run in Jupyter Notebook for a better interactive experience

## Technical Requirements

- Python 3.7+
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- statsmodels >= 0.13.0
- scipy >= 1.7.0

## Contact

For questions or suggestions, please contact via GitHub Issues.
