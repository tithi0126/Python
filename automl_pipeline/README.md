# AutoML - Complete Machine Learning Library

A comprehensive ML library that automates the entire pipeline:
**Data → EDA → Preprocessing → Split → Model → Evaluate → Predict**

## Installation

```bash
pip install automl-quickpipe
```

## Quick Start

```python
import pandas as pd
from automl import AutoML

# Load your dataset
df = pd.read_csv('your_data.csv')

# Initialize and run complete analysis
ml = AutoML(df, target_col='target_column')
ml.run_complete_analysis()

# Make predictions
new_data = [features]
ml.predict_new(new_data)
```

## Features

- **Automated EDA**: Statistical summaries, missing value detection, correlation matrices, and distribution plots.
- **Smart Preprocessing**: Automatic handling of missing values, label encoding for categorical data, and feature scaling.
- **Problem Detection**: Automatically detects whether to use Classification or Regression based on the target column.
- **Multiple Algorithms**: Trains and compares multiple models (KNN, Decision Trees, Random Forest, SVM, etc.).
- **Model Evaluation**: Comprehensive metrics (Accuracy, F1-Score, R², RMSE) with cross-validation.
- **Best Model Selection**: Automatically identifies and uses the best-performing model for predictions.

## Requirements

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- ipython (for Jupyter notebook support)

## License

MIT
