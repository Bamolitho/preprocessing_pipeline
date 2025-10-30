# Professional Data Preprocessing Pipeline

A comprehensive, modular, and production-ready preprocessing pipeline for machine learning projects in Python.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Steps](#pipeline-steps)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Model-Specific Recommendations](#model-specific-recommendations)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)

---

## Overview

This preprocessing pipeline provides a complete, professional-grade solution for preparing data for machine learning models. It handles all common preprocessing tasks including data cleaning, missing value imputation, encoding, scaling, dimensionality reduction, and feature selection.

**Key Characteristics:**
- **Modular & Configurable**: All steps controlled via YAML/JSON configuration
- **Production-Ready**: Save/load pipelines with joblib for deployment
- **Well-Documented**: Every line is commented and explained
- **Model-Agnostic**: Adapt preprocessing based on your model type
- **Comprehensive**: 8 complete preprocessing steps from cleaning to feature selection

---

## Features

### Core Capabilities

1. **Data Cleaning**
   - Duplicate row removal
   - Outlier detection (IQR and Z-score methods)

2. **Missing Value Handling**
   - Numerical imputation (mean, median, most_frequent)
   - Categorical imputation (most_frequent, constant)

3. **Categorical Encoding**
   - One-Hot Encoding (nominal variables)
   - Label Encoding (ordinal variables)

4. **Statistical Filtering**
   - Low variance feature removal
   - Highly correlated feature removal

5. **Scaling/Normalization**
   - StandardScaler (z-score normalization)
   - MinMaxScaler (0-1 range normalization)

6. **Dimensionality Reduction**
   - PCA (Principal Component Analysis)
   - ICA (Independent Component Analysis)
   - LDA (Linear Discriminant Analysis)

7. **Feature Selection**
   - ANOVA F-test
   - Mutual Information
   - Chi-squared test
   - Random Forest importance

8. **Pipeline Management**
   - Save fitted pipelines
   - Load and reuse pipelines
   - Transform new data consistently

---

## Installation

### Requirements

```bash
pip install --upgrade pip && pip install -r requirements.txt
```

### Python Version

- Python 3.7+

---

## Quick Start

### 1. Basic Usage

```python
from preprocessing_pipeline import PreprocessingPipeline
import pandas as pd

# Load your data
X_train = pd.read_csv('train_data.csv')
y_train = pd.read_csv('train_labels.csv')

# Initialize pipeline with configuration
pipeline = PreprocessingPipeline(config_path='config.yaml')

# Fit and transform training data
X_train_processed, y_train = pipeline.fit_transform(X_train, y_train)

# Transform test data
X_test_processed = pipeline.transform(X_test)

# Save pipeline for later use
pipeline.save_pipeline('my_pipeline.pkl')
```

### 2. Loading a Saved Pipeline

```python
# Load previously saved pipeline
pipeline = PreprocessingPipeline.load_pipeline('my_pipeline.pkl')

# Transform new data
X_new_processed = pipeline.transform(X_new)
```

---

## Pipeline Steps

### Visual Flow

```
┌─────────────────────────────────────┐
│  [1] DATA CLEANING                  │
│  - Remove duplicates                │
│  - Handle outliers (IQR/Z-score)    │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  [2] MISSING VALUES HANDLING        │
│  - Numerical imputation             │
│  - Categorical imputation           │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  [3] COLUMN TYPE IDENTIFICATION     │
│  - Identify numerical columns       │
│  - Identify categorical columns     │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  [4] CATEGORICAL ENCODING           │
│  - One-Hot Encoding                 │
│  - Label Encoding                   │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  [5] STATISTICAL FILTERING          │
│  - Remove low variance features     │
│  - Remove correlated features       │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  [6] SCALING/NORMALIZATION          │
│  - StandardScaler (z-score)         │
│  - MinMaxScaler (0-1 range)         │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  [7] DIMENSIONALITY REDUCTION       │
│  - PCA (variance maximization)      │
│  - ICA (independent sources)        │
│  - LDA (class separation)           │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  [8] FEATURE SELECTION              │
│  - Statistical tests (F-test, χ²)   │
│  - Information theory (MI)          │
│  - Model-based (Random Forest)      │
└─────────────────────────────────────┘
```

---

## Configuration

### Configuration File Structure (config.yaml)

```yaml
preprocessing:
  # Data Cleaning
  remove_duplicates: true
  handle_outliers: false
  outlier_method: 'iqr'  # or 'zscore'
  
  # Missing Values
  handle_missing: true
  missing_numerical_strategy: 'mean'  # 'mean', 'median', 'most_frequent'
  missing_categorical_strategy: 'most_frequent'
  
  # Categorical Encoding
  encode_categorical: true
  categorical_encoding_method: 'onehot'  # 'onehot' or 'label'
  
  # Statistical Filtering
  remove_low_variance: true
  variance_threshold: 0.01
  remove_correlated: true
  correlation_threshold: 0.95
  
  # Scaling
  use_scaler: true
  scaling_method: 'standard'  # 'standard' or 'minmax'
  
  # Dimensionality Reduction (choose ONE)
  use_pca: false
  n_components: 10
  use_ica: false
  ica_components: 10
  use_lda: false
  lda_components: 'auto'
  
  # Feature Selection
  feature_selection: false
  selection_method: 'auto'  # 'auto', 'f_test', 'mutual_info', 'chi2', 'random_forest'
  k_best: 10
  problem_type: 'auto'  # 'classification', 'regression', or 'auto'
```

### Key Configuration Parameters

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `remove_duplicates` | bool | true/false | Remove duplicate rows |
| `handle_outliers` | bool | true/false | Detect and remove outliers |
| `outlier_method` | str | 'iqr', 'zscore' | Method for outlier detection |
| `handle_missing` | bool | true/false | Impute missing values |
| `missing_numerical_strategy` | str | 'mean', 'median', 'most_frequent' | Strategy for numerical columns |
| `use_scaler` | bool | true/false | Apply feature scaling |
| `scaling_method` | str | 'standard', 'minmax' | Scaling algorithm |
| `use_pca` | bool | true/false | Apply PCA |
| `n_components` | int/str | integer or 'auto' | Number of PCA components |
| `feature_selection` | bool | true/false | Enable feature selection |
| `selection_method` | str | 'auto', 'f_test', etc. | Feature selection method |

---

## Usage Examples

### Example 1: Basic Preprocessing

```python
from preprocessing_pipeline import PreprocessingPipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize and fit pipeline
pipeline = PreprocessingPipeline('config.yaml')
X_train_proc, y_train = pipeline.fit_transform(X_train, y_train)
X_test_proc = pipeline.transform(X_test)
```

### Example 2: With PCA Dimensionality Reduction

```python
import yaml

# Load and modify config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Enable PCA
config['preprocessing']['use_pca'] = True
config['preprocessing']['n_components'] = 5
config['preprocessing']['use_scaler'] = True  # Required before PCA

# Save modified config
with open('config_pca.yaml', 'w') as f:
    yaml.dump(config, f)

# Use modified config
pipeline = PreprocessingPipeline('config_pca.yaml')
X_proc, y = pipeline.fit_transform(X, y)
```

### Example 3: Feature Selection

```python
# Configure for feature selection
config['preprocessing']['feature_selection'] = True
config['preprocessing']['selection_method'] = 'random_forest'
config['preprocessing']['k_best'] = 15

# Initialize pipeline
pipeline = PreprocessingPipeline('config_selection.yaml')
X_proc, y = pipeline.fit_transform(X, y)

# Get selected feature names
selected_features = pipeline.get_feature_names()
print(f"Selected features: {selected_features}")
```

---

## Model-Specific Recommendations

### Linear Models (Linear/Logistic Regression, SVM)

```yaml
preprocessing:
  use_scaler: true
  scaling_method: 'standard'
  encode_categorical: true
  categorical_encoding_method: 'onehot'
  feature_selection: true
  remove_correlated: true
```

**Why:** Linear models are sensitive to feature scale and benefit from decorrelated features.

### Tree-Based Models (Random Forest, XGBoost)

```yaml
preprocessing:
  use_scaler: false  # Trees are scale-invariant
  encode_categorical: true
  categorical_encoding_method: 'label'  # Label encoding is sufficient
  remove_correlated: false  # Trees handle correlation naturally
  feature_selection: false  # Trees perform implicit feature selection
```

**Why:** Tree-based models make splits based on thresholds, so scaling doesn't affect them.

### Distance-Based Models (KNN, K-Means)

```yaml
preprocessing:
  use_scaler: true  # CRITICAL!
  scaling_method: 'minmax'  # or 'standard'
  encode_categorical: true
  categorical_encoding_method: 'onehot'
  remove_correlated: true
```

**Why:** Distance calculations are highly sensitive to feature scales.

### Neural Networks (Deep Learning)

```yaml
preprocessing:
  use_scaler: true
  scaling_method: 'standard'  # or 'minmax'
  encode_categorical: true
  categorical_encoding_method: 'onehot'
  feature_selection: false  # Networks learn feature importance
```

**Why:** Neural networks converge faster with normalized inputs.

---

## API Reference

### PreprocessingPipeline Class

#### `__init__(config_path: str)`

Initialize the preprocessing pipeline.

**Parameters:**
- `config_path` (str): Path to YAML or JSON configuration file

**Example:**
```python
pipeline = PreprocessingPipeline('config.yaml')
```

#### `fit_transform(X, y=None, target_column=None)`

Fit the pipeline and transform the data.

**Parameters:**
- `X` (pd.DataFrame): Input features
- `y` (pd.Series, optional): Target variable
- `target_column` (str, optional): Name of target column if in X

**Returns:**
- `Tuple[np.ndarray, Optional[np.ndarray]]`: Transformed features and target

**Example:**
```python
X_proc, y = pipeline.fit_transform(X_train, y_train)
```

#### `transform(X, target_column=None)`

Transform new data using fitted pipeline.

**Parameters:**
- `X` (pd.DataFrame): Input features to transform
- `target_column` (str, optional): Name of target column if in X

**Returns:**
- `np.ndarray`: Transformed features

**Example:**
```python
X_test_proc = pipeline.transform(X_test)
```

#### `save_pipeline(filepath: str)`

Save the fitted pipeline to disk.

**Parameters:**
- `filepath` (str): Path where to save the pipeline

**Example:**
```python
pipeline.save_pipeline('my_pipeline.pkl')
```

#### `load_pipeline(filepath: str)` [classmethod]

Load a saved pipeline from disk.

**Parameters:**
- `filepath` (str): Path to saved pipeline

**Returns:**
- `PreprocessingPipeline`: Loaded pipeline instance

**Example:**
```python
pipeline = PreprocessingPipeline.load_pipeline('my_pipeline.pkl')
```

#### `get_feature_names()`

Get output feature names after preprocessing.

**Returns:**
- `List[str]`: List of feature names

#### `get_config()`

Get the current configuration.

**Returns:**
- `Dict[str, Any]`: Configuration dictionary

#### `print_summary()`

Print a summary of the preprocessing pipeline including all fitted transformers.

---

## Project Structure

```
preprocessing_pipeline/
│
├── preprocessing_pipeline.py    # Main pipeline class
├── config.yaml                  # Default configuration file
├── main.py                      # Usage examples and demonstrations
├── README.md                    # This documentation
│
├── saved_models/                # Directory for saved pipelines (created at runtime)
│   ├── pipeline.pkl
│   └── model.pkl
│
└── data/                        # Directory for datasets (user-provided)
    ├── train.csv
    └── test.csv
```

---

## Detailed Step Explanations

### Step 1: Data Cleaning

**Purpose:** Remove noise and inconsistencies from the dataset.

**Operations:**
- **Duplicate Removal:** Identifies and removes exact duplicate rows that could bias model training
- **Outlier Detection:** 
  - **IQR Method:** Removes values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
  - **Z-score Method:** Removes values with |z-score| > threshold (default: 3)

**When to use:**
- Always use duplicate removal
- Use outlier detection cautiously - outliers may contain important information
- Consider domain knowledge before removing outliers

### Step 2: Missing Values Handling

**Purpose:** Impute missing values to avoid dropping valuable samples.

**Strategies:**
- **Numerical:**
  - `mean`: Good for normally distributed data
  - `median`: Robust to outliers, good for skewed distributions
  - `most_frequent`: For discrete numerical values
- **Categorical:**
  - `most_frequent`: Fill with the mode (most common value)
  - `constant`: Fill with a custom value (e.g., "Unknown")

**Best Practices:**
- Use `median` for numerical data with outliers
- Use `mean` for normally distributed data
- Always use `most_frequent` for categorical data

### Step 3: Column Type Identification

**Purpose:** Automatically detect numerical and categorical columns.

**Detection Logic:**
- Numerical: `int64`, `float64` dtypes
- Categorical: `object`, `category` dtypes

**Note:** This step is automatic and doesn't require configuration.

### Step 4: Categorical Encoding

**Purpose:** Convert categorical variables to numerical format.

**Methods:**
- **One-Hot Encoding:**
  - Creates binary columns for each category
  - Use for: Nominal variables (no order)
  - Example: Colors (Red, Blue, Green) → 3 binary columns
  - **Warning:** Can create many features (curse of dimensionality)

- **Label Encoding:**
  - Assigns integer labels to categories
  - Use for: Ordinal variables (natural order)
  - Example: Education (High School=0, Bachelor=1, Master=2, PhD=3)
  - **Warning:** Implies ordering, use only for ordinal data

**Recommendation:**
- Default to One-Hot for most cases
- Use Label Encoding only when order matters

### Step 5: Statistical Filtering

**Purpose:** Remove redundant or uninformative features.

**Operations:**

1. **Low Variance Removal:**
   - Removes features with variance below threshold
   - Rationale: Low variance = little information
   - Formula: Var(X) < threshold → remove X
   - Typical threshold: 0.01

2. **Correlation Filtering:**
   - Removes highly correlated feature pairs
   - Rationale: Correlated features are redundant
   - Formula: |Corr(Xi, Xj)| > threshold → remove one
   - Typical threshold: 0.95

**When to apply:**
- Always for linear models (reduces multicollinearity)
- Optional for tree-based models (they handle correlation)

### Step 6: Scaling/Normalization

**Purpose:** Transform features to similar scales.

**Methods:**

1. **StandardScaler (Z-score normalization):**
   - Formula: z = (x - μ) / σ
   - Result: mean = 0, std = 1
   - Use for: Most ML algorithms, especially when features have different units
   - Example: Age (0-100) and Income (0-1M) → both scaled to similar range

2. **MinMaxScaler (Min-Max normalization):**
   - Formula: x_scaled = (x - min) / (max - min)
   - Result: range [0, 1]
   - Use for: Neural networks, algorithms that need bounded inputs
   - **Warning:** Sensitive to outliers

**Critical Notes:**
- ✅ **Always scale before PCA/ICA/LDA**
- ✅ **Always scale for SVM, KNN, Neural Networks**
- ❌ **Never scale for tree-based models** (Random Forest, XGBoost)

### Step 7: Dimensionality Reduction

**Purpose:** Reduce feature count while preserving information.

**Methods:**

1. **PCA (Principal Component Analysis) - Unsupervised:**
   - **Goal:** Maximize variance
   - **Method:** Linear transformation to orthogonal components
   - **Formula:** X_reduced = X · V, where V are eigenvectors
   - **Use cases:**
     - High-dimensional data (e.g., images, text)
     - Visualization (reduce to 2-3 dimensions)
     - Speed up training
   - **Pros:** Simple, interpretable variance
   - **Cons:** Linear, sensitive to scaling

2. **ICA (Independent Component Analysis) - Unsupervised:**
   - **Goal:** Separate independent sources
   - **Method:** Find statistically independent components
   - **Use cases:**
     - Signal separation (e.g., audio, EEG)
     - Mixed source problems
   - **Pros:** Finds independent sources
   - **Cons:** More complex, requires more assumptions

3. **LDA (Linear Discriminant Analysis) - Supervised:**
   - **Goal:** Maximize class separability
   - **Method:** Find directions that separate classes
   - **Formula:** Maximize between-class variance / within-class variance
   - **Use cases:**
     - Classification problems
     - When you want class-aware dimensionality reduction
   - **Pros:** Supervised, maximizes discrimination
   - **Cons:** Limited to (n_classes - 1) components, requires labels
   - **Note:** ✅ **Best for classification when you have labels**

**Selection Guide:**
- Use **PCA** for general dimensionality reduction
- Use **ICA** for signal separation problems
- Use **LDA** for classification (requires labels)
- **Only use ONE method at a time**

### Step 8: Feature Selection

**Purpose:** Select the most relevant features for prediction.

**Methods:**

1. **F-test (ANOVA):**
   - Tests statistical significance of features
   - Classification: f_classif
   - Regression: f_regression
   - Fast and effective

2. **Mutual Information:**
   - Measures dependence between feature and target
   - Captures non-linear relationships
   - More computationally expensive

3. **Chi-squared (χ²):**
   - Tests independence for categorical data
   - Classification only
   - Requires non-negative features

4. **Random Forest Importance:**
   - Uses feature importance from Random Forest
   - Captures non-linear relationships
   - Most robust but slowest

**Selection Strategy:**
- Use `auto` to let the pipeline choose
- Use `f_test` for quick, linear relationships
- Use `mutual_info` for non-linear relationships
- Use `random_forest` for best results (slower)

---

## Best Practices

### 1. Configuration Management

```python
# Bad: Hardcoding parameters
pipeline.variance_threshold = 0.01
pipeline.use_pca = True

# Good: Using configuration files
pipeline = PreprocessingPipeline('config.yaml')
```

### 2. Pipeline Persistence

```python
# Always save your fitted pipeline
pipeline.fit_transform(X_train, y_train)
pipeline.save_pipeline('production_pipeline.pkl')

# Load in production
prod_pipeline = PreprocessingPipeline.load_pipeline('production_pipeline.pkl')
X_new_processed = prod_pipeline.transform(X_new)
```

### 3. Train/Test Consistency

```python
# Bad: Fitting on test data
pipeline.fit_transform(X_test)  # NEVER DO THIS

# Good: Only transform test data
pipeline.fit_transform(X_train, y_train)  # Fit on train
X_test_processed = pipeline.transform(X_test)  # Transform test
```

### 4. Feature Leakage Prevention

```python
# Bad: Including target in features
X_with_target = pd.concat([X, y], axis=1)
pipeline.fit_transform(X_with_target)

# Good: Separate features and target
pipeline.fit_transform(X, y)
```

### 5. Handling New Categories

```python
# When using One-Hot Encoding, new categories in test data
# will be ignored automatically. Consider this in production.

# To handle explicitly:
# - Use Label Encoding with handle_unknown='ignore'
# - Or ensure all categories are in training data
```

---

## Troubleshooting

### Issue 1: "ValueError: could not convert string to float"

**Cause:** Categorical variables not encoded

**Solution:**
```yaml
preprocessing:
  encode_categorical: true
```

### Issue 2: "All features have zero variance"

**Cause:** Variance threshold too high or constant features

**Solution:**
```yaml
preprocessing:
  variance_threshold: 0.0001  # Lower threshold
```

### Issue 3: "LDA requires target variable"

**Cause:** Using LDA without providing y

**Solution:**
```python
# Always provide target for LDA
X_proc, y = pipeline.fit_transform(X, y)  # ✓ Correct
X_proc = pipeline.fit_transform(X)  # ✗ Wrong for LDA
```

### Issue 4: Poor model performance after preprocessing

**Possible causes:**
1. **Over-aggressive filtering:** Too many features removed
   - Lower `variance_threshold` and `correlation_threshold`
2. **Wrong scaling for model type:** Scaling tree-based models
   - Set `use_scaler: false` for Random Forest, XGBoost
3. **Information loss from dimensionality reduction:** Too few components
   - Increase `n_components` or disable DR

---

## Performance Considerations

### Memory Usage

- **One-Hot Encoding:** Can significantly increase memory for high-cardinality categorical variables
  - Example: 1000 unique categories → 1000 new columns
  - Solution: Use Label Encoding or target encoding

- **Large Datasets:** Process in chunks if memory is limited
  ```python
  # For very large datasets, consider using Dask or chunked processing
  chunk_size = 10000
  for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
      X_processed = pipeline.transform(chunk)
  ```

### Speed Optimization

- **Feature Selection:** Random Forest importance is slowest but most accurate
  - Fast: `f_test` (milliseconds)
  - Medium: `mutual_info` (seconds)
  - Slow: `random_forest` (minutes for large datasets)

- **Dimensionality Reduction:**
  - PCA: O(min(n_samples², n_features²))
  - ICA: O(n_iterations × n_samples × n_components²)
  - LDA: O(n_samples × n_features × n_classes)

---

## Testing the Pipeline

### Unit Test Example

```python
import unittest
import pandas as pd
import numpy as np
from preprocessing_pipeline import PreprocessingPipeline

class TestPreprocessingPipeline(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        self.X = pd.DataFrame({
            'num1': [1, 2, 3, 4, 5],
            'num2': [10, 20, 30, 40, 50],
            'cat1': ['A', 'B', 'A', 'B', 'A']
        })
        self.y = pd.Series([0, 1, 0, 1, 0])
    
    def test_basic_transform(self):
        """Test basic transformation"""
        pipeline = PreprocessingPipeline('config.yaml')
        X_proc, y_proc = pipeline.fit_transform(self.X, self.y)
        
        self.assertIsNotNone(X_proc)
        self.assertEqual(len(X_proc), len(self.X))
    
    def test_save_load(self):
        """Test pipeline persistence"""
        pipeline = PreprocessingPipeline('config.yaml')
        pipeline.fit_transform(self.X, self.y)
        
        pipeline.save_pipeline('test_pipeline.pkl')
        loaded_pipeline = PreprocessingPipeline.load_pipeline('test_pipeline.pkl')
        
        X_proc1 = pipeline.transform(self.X)
        X_proc2 = loaded_pipeline.transform(self.X)
        
        np.testing.assert_array_almost_equal(X_proc1, X_proc2)

if __name__ == '__main__':
    unittest.main()
```

---

## Logging and Monitoring

The pipeline provides comprehensive logging with the following markers:

- `[INFO]`: Informational messages about current operations
- `[OK]`: Successful completion of operations
- `[WARNING]`: Non-critical issues that may need attention
- `[FAILED]`: Critical errors that stop execution

Example output:
```
[INFO] Loading configuration...
[OK] Configuration loaded successfully
[INFO] Checking for duplicate rows...
[OK] No duplicate rows found
[WARNING] Found 15 missing values
[INFO] Imputing missing values...
[OK] All missing values handled successfully
```

---

## Contributing

This is a professional template. Feel free to extend it with:

1. **Additional encoders:** Target encoding, Binary encoding
2. **More imputation strategies:** KNN imputation, MICE
3. **Advanced feature selection:** LASSO, Elastic Net, Boruta
4. **Custom transformers:** Domain-specific transformations
5. **Automated hyperparameter tuning:** GridSearch for preprocessing params

---

## Acknowledgments

Built using:
- **scikit-learn**: Machine learning library
- **pandas**: Data manipulation
- **NumPy**: Numerical computing

---

## Support

For issues, questions, or suggestions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review the [API Reference](#api-reference)
3. Run the examples in `main.py`

---

## Summary

This preprocessing pipeline provides:

✅ **8 comprehensive preprocessing steps**  

✅ **Fully configurable via YAML/JSON**  

✅ **Production-ready with save/load functionality**  

✅ **Model-specific recommendations**  

✅ **Extensive documentation and examples** 
✅ **Professional logging and error handling** 
✅ **PEP8 compliant code**  

**Use it to jump-start your machine learning projects with professional-grade data preprocessing!**