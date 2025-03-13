# Parameter Clustering for Technical Trading Strategy Optimization

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Requirements](#requirements)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Features](#features)
- [Author](#author)

## Introduction

This repository focuses on implementing clustering techniques to optimize parameters for technical trading strategies. The `ParameterClusterer` class uses various clustering methods (K-means, GMM, Agglomerative) to identify optimal parameter combinations based on backtest metrics.

The main goal is to group similar parameter configurations and select the most effective ones based on their performance metrics. This approach helps:
- Identify robust parameter combinations
- Reduce parameter space complexity
- Optimize trading strategy performance

## Installation

### Requirements

The project requires Python 3.11.9 and the key dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Here's a basic example of how to use the `ParameterClusterer`:

```python
from search_parameters import ParameterClusterer

# Configure the clusterer
config = {
    'backtest_metrics': ['sharpe_ratio', 'max_drawdown', 'profit_factor'],
    'weights_metrics': [0.4, 0.3, 0.3],
    'pct_sample_cluster': 5,
    'method_cluster': 'kmeans'
}

# Load the backtest results
stats_params_backtest = pd.read_csv('stats_backtest.csv')

# Initialize the clusterer
clusterer = ParameterClusterer(config)

# Find best parameters
best_params = clusterer.find_best_parameters(stats_params_backtest)
```

## Repository Structure

```
.
│   README.md
│   search_parameters.py
│   requirements.txt
│   function_cluster.py
│   main.py
│   plots.py
```

## Features

- **Multiple Clustering Methods**: Supports K-means, GMM, and Agglomerative clustering
- **Flexible Configuration**: Easy parameter configuration through dictionary input
- **Data Preprocessing**: 
  - Sigmoid normalization
  - Zero-value filtering
  - NaN handling
- **Cluster Evaluation**:
  - Intra-cluster variability analysis
  - Pairwise distance metrics
  - Sample size validation
- **Parameter Optimization**:
  - Weighted metric evaluation
  - Best cluster selection
  - Median parameter extraction

## Author

Jhon Jairo Realpe

jhon.td.03@gmail.com
