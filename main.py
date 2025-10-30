"""
Main Example Script - Preprocessing Pipeline Demonstration
===========================================================
This script demonstrates how to use the preprocessing pipeline
for both training and inference.
"""
import yaml
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Import our custom preprocessing pipeline
from preprocessing_pipeline import PreprocessingPipeline


# Directory structure
OUTPUT_DIR = Path('outputs')
PIPELINE_DIR = OUTPUT_DIR / 'pipelines'
MODEL_DIR = OUTPUT_DIR / 'models'
CONFIG_DIR = Path('config')


def example_1_basic_usage():
    """
    Example 1: Basic usage with Iris dataset
    =========================================
    Demonstrates the simplest way to use the pipeline.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: BASIC USAGE WITH IRIS DATASET")
    print("="*80 + "\n")
    
    # -------------------------------------------------------------------------
    # Step 1: Load the data
    # -------------------------------------------------------------------------
    print("[INFO] Loading Iris dataset...")
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    
    print(f"[OK] Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # -------------------------------------------------------------------------
    # Step 2: Split into train and test sets
    # -------------------------------------------------------------------------
    print("[INFO] Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[OK] Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

    # -------------------------------------------------------------------------
    # Step 3: Initialize and fit the preprocessing pipeline
    # -------------------------------------------------------------------------
    print("[INFO] Initializing preprocessing pipeline...")
    pipeline = PreprocessingPipeline(config_path='config.yaml')

    # Fit and transform training data
    X_train_processed, y_train = pipeline.fit_transform(X_train, y_train)

    # Transform test data using the fitted pipeline
    X_test_processed = pipeline.transform(X_test)

    
    # -------------------------------------------------------------------------
    # Step 4: Train a model on preprocessed data
    # -------------------------------------------------------------------------
    print("[INFO] Training Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_processed, y_train)
    print("[OK] Model training complete")
    
    # -------------------------------------------------------------------------
    # Step 5: Evaluate the model
    # -------------------------------------------------------------------------
    print("[INFO] Evaluating model on test set...")
    y_pred = model.predict(X_test_processed)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n[OK] Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # -------------------------------------------------------------------------
    # Step 6: Save the pipeline for later use
    # -------------------------------------------------------------------------
    print("[INFO] Saving preprocessing pipeline...")
    pipeline.save_pipeline(str(PIPELINE_DIR / 'saved_pipeline.pkl'))
    
    print("\n" + "="*80)
    print("[OK] EXAMPLE 1 COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")


def example_2_with_pca():
    """
    Example 2: Using PCA for dimensionality reduction
    ==================================================
    Demonstrates how to use PCA to reduce feature dimensions.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: USING PCA FOR DIMENSIONALITY REDUCTION")
    print("="*80 + "\n")
    
    # -------------------------------------------------------------------------
    # Step 1: Load a high-dimensional dataset
    # -------------------------------------------------------------------------
    print("[INFO] Loading Breast Cancer dataset...")
    cancer = load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = pd.Series(cancer.target, name='target')
    
    print(f"[OK] Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # -------------------------------------------------------------------------
    # Step 2: Split the data
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # -------------------------------------------------------------------------
    # Step 3: Modify configuration to enable PCA
    # -------------------------------------------------------------------------
    print("[INFO] Initializing pipeline with PCA enabled...")
    
    # Load and modify config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Enable PCA and set components
    config['preprocessing']['use_pca'] = True
    config['preprocessing']['n_components'] = 10
    config['preprocessing']['use_scaler'] = True  # Important: scale before PCA
    
    # Save modified config
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_DIR / 'config_pca.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Initialize pipeline with PCA config
    pipeline = PreprocessingPipeline(config_path=str(CONFIG_DIR / 'config_pca.yaml'))
    
    # -------------------------------------------------------------------------
    # Step 4: Fit and transform
    # -------------------------------------------------------------------------
    X_train_processed, y_train = pipeline.fit_transform(X_train, y_train)
    X_test_processed = pipeline.transform(X_test)
    
    print(f"[INFO] Features reduced from {X_train.shape[1]} to {X_train_processed.shape[1]}")
    
    # -------------------------------------------------------------------------
    # Step 5: Train and evaluate
    # -------------------------------------------------------------------------
    print("[INFO] Training SVM classifier...")
    model = SVC(kernel='rbf', random_state=42)
    model.fit(X_train_processed, y_train)
    
    y_pred = model.predict(X_test_processed)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n[OK] Model Accuracy with PCA: {accuracy:.4f}")
    
    print("\n" + "="*80)
    print("[OK] EXAMPLE 2 COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")


def example_3_loading_saved_pipeline():
    """
    Example 3: Loading and using a saved pipeline
    ==============================================
    Demonstrates how to load a previously saved pipeline.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: LOADING AND USING A SAVED PIPELINE")
    print("="*80 + "\n")
    
    # -------------------------------------------------------------------------
    # Step 1: Load the saved pipeline
    # -------------------------------------------------------------------------
    print("[INFO] Loading saved pipeline from disk...")
    
    try:
        pipeline = PreprocessingPipeline.load_pipeline(str(PIPELINE_DIR / 'saved_pipeline.pkl'))
        
        # -------------------------------------------------------------------------
        # Step 2: Prepare new data
        # -------------------------------------------------------------------------
        print("[INFO] Preparing new data for prediction...")
        
        # Load new data (simulating new samples)
        iris = load_iris()
        X_new = pd.DataFrame(iris.data[:5], columns=iris.feature_names)
        
        print("[INFO] New data shape:", X_new.shape)
        print("\nNew data (first 5 samples):")
        print(X_new)
        
        # -------------------------------------------------------------------------
        # Step 3: Transform new data
        # -------------------------------------------------------------------------
        print("\n[INFO] Transforming new data with saved pipeline...")
        X_new_processed = pipeline.transform(X_new)
        
        print(f"[OK] New data transformed: {X_new_processed.shape}")
        print("\nProcessed features:")
        print(X_new_processed)
        
        # -------------------------------------------------------------------------
        # Step 4: Print pipeline summary
        # -------------------------------------------------------------------------
        pipeline.print_summary()
        
        print("\n" + "="*80)
        print("[OK] EXAMPLE 3 COMPLETED SUCCESSFULLY")
        print("="*80 + "\n")
        
    except FileNotFoundError:
        print("[WARNING] Saved pipeline not found. Please run Example 1 first.")
        print("="*80 + "\n")


def example_4_feature_selection():
    """
    Example 4: Using feature selection
    ===================================
    Demonstrates how to select the most important features.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: FEATURE SELECTION")
    print("="*80 + "\n")
    
    # -------------------------------------------------------------------------
    # Step 1: Load data
    # -------------------------------------------------------------------------
    print("[INFO] Loading Breast Cancer dataset...")
    cancer = load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = pd.Series(cancer.target, name='target')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # -------------------------------------------------------------------------
    # Step 2: Configure pipeline with feature selection
    # -------------------------------------------------------------------------
    print("[INFO] Configuring pipeline with feature selection...")
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Enable feature selection
    config['preprocessing']['feature_selection'] = True
    config['preprocessing']['selection_method'] = 'random_forest'
    config['preprocessing']['k_best'] = 15
    config['preprocessing']['use_scaler'] = True
    
    # Save modified config
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_DIR / 'config_feature_selection.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Initialize pipeline
    pipeline = PreprocessingPipeline(config_path=str(CONFIG_DIR / 'config_feature_selection.yaml'))
    
    # -------------------------------------------------------------------------
    # Step 3: Fit and transform with feature selection
    # -------------------------------------------------------------------------
    X_train_processed, y_train = pipeline.fit_transform(X_train, y_train)
    X_test_processed = pipeline.transform(X_test)
    
    print(f"[INFO] Features selected: {X_train_processed.shape[1]} out of {X_train.shape[1]}")
    print(f"[INFO] Selected feature names: {pipeline.get_feature_names()}")
    
    # -------------------------------------------------------------------------
    # Step 4: Train and evaluate
    # -------------------------------------------------------------------------
    print("[INFO] Training classifier with selected features...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_processed, y_train)
    
    y_pred = model.predict(X_test_processed)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n[OK] Model Accuracy with feature selection: {accuracy:.4f}")
    
    print("\n" + "="*80)
    print("[OK] EXAMPLE 4 COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")


def example_5_complete_workflow():
    """
    Example 5: Complete ML workflow with preprocessing
    ===================================================
    Demonstrates a complete machine learning workflow including:
    - Data loading and exploration
    - Preprocessing with all steps
    - Model training and evaluation
    - Pipeline saving and loading
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: COMPLETE MACHINE LEARNING WORKFLOW")
    print("="*80 + "\n")
    
    # -------------------------------------------------------------------------
    # Step 1: Load and explore data
    # -------------------------------------------------------------------------
    print("[STEP 1] DATA LOADING AND EXPLORATION")
    print("-"*80)
    
    cancer = load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = pd.Series(cancer.target, name='target')
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"Missing values: {X.isnull().sum().sum()}")
    print(f"Duplicate rows: {X.duplicated().sum()}")
    
    # -------------------------------------------------------------------------
    # Step 2: Data splitting
    # -------------------------------------------------------------------------
    print("\n[STEP 2] DATA SPLITTING")
    print("-"*80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape[0]} samples ({y_train.sum()} positive)")
    print(f"Test set: {X_test.shape[0]} samples ({y_test.sum()} positive)")
    
    # -------------------------------------------------------------------------
    # Step 3: Configure comprehensive preprocessing
    # -------------------------------------------------------------------------
    print("\n[STEP 3] PREPROCESSING CONFIGURATION")
    print("-"*80)
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Configure for optimal preprocessing
    config['preprocessing']['remove_duplicates'] = True
    config['preprocessing']['handle_missing'] = True
    config['preprocessing']['encode_categorical'] = True
    config['preprocessing']['remove_low_variance'] = True
    config['preprocessing']['variance_threshold'] = 0.01
    config['preprocessing']['remove_correlated'] = True
    config['preprocessing']['correlation_threshold'] = 0.95
    config['preprocessing']['use_scaler'] = True
    config['preprocessing']['scaling_method'] = 'standard'
    config['preprocessing']['feature_selection'] = True
    config['preprocessing']['selection_method'] = 'f_test'
    config['preprocessing']['k_best'] = 20
    
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_DIR / 'config_complete.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print("[OK] Configuration set for comprehensive preprocessing")
    
    # -------------------------------------------------------------------------
    # Step 4: Fit preprocessing pipeline
    # -------------------------------------------------------------------------
    print("\n[STEP 4] FITTING PREPROCESSING PIPELINE")
    print("-"*80)
    
    pipeline = PreprocessingPipeline(config_path=str(CONFIG_DIR / 'config_complete.yaml'))
    X_train_processed, y_train = pipeline.fit_transform(X_train, y_train)
    
    # -------------------------------------------------------------------------
    # Step 5: Transform test data
    # -------------------------------------------------------------------------
    print("\n[STEP 5] TRANSFORMING TEST DATA")
    print("-"*80)
    
    X_test_processed = pipeline.transform(X_test)
    
    print(f"[INFO] Original features: {X_train.shape[1]}")
    print(f"[INFO] Processed features: {X_train_processed.shape[1]}")
    print(f"[INFO] Reduction: {(1 - X_train_processed.shape[1]/X_train.shape[1])*100:.1f}%")
    
    # -------------------------------------------------------------------------
    # Step 6: Model training and comparison
    # -------------------------------------------------------------------------
    print("\n[STEP 6] MODEL TRAINING AND EVALUATION")
    print("-"*80)
    
    # Train multiple models for comparison
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n[INFO] Training {name}...")
        model.fit(X_train_processed, y_train)
        
        # Evaluate
        y_pred_train = model.predict(X_train_processed)
        y_pred_test = model.predict(X_test_processed)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        results[name] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc
        }
        
        print(f"[OK] {name} - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    # -------------------------------------------------------------------------
    # Step 7: Save pipeline and models
    # -------------------------------------------------------------------------
    print("\n[STEP 7] SAVING PIPELINE AND MODELS")
    print("-"*80)
    
    pipeline.save_pipeline(str(PIPELINE_DIR / 'complete_pipeline.pkl'))
    
    for name, model in models.items():
        filename = MODEL_DIR / f"model_{name.replace(' ', '_').lower()}.pkl"
        joblib.dump(model, filename)
        print(f"[OK] Saved {name} to {filename}")
    
    # -------------------------------------------------------------------------
    # Step 8: Print final summary
    # -------------------------------------------------------------------------
    print("\n[STEP 8] FINAL SUMMARY")
    print("-"*80)
    
    print("\nModel Performance Comparison:")
    print("-"*40)
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"  Test Accuracy:  {metrics['test_accuracy']:.4f}")
        print(f"  Overfitting:    {(metrics['train_accuracy'] - metrics['test_accuracy'])*100:.2f}%")
    
    pipeline.print_summary()
    
    print("\n" + "="*80)
    print("[OK] EXAMPLE 5 COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")


def example_6_custom_dataset_with_missing_values():
    """
    Example 6: Handling a custom dataset with missing values and mixed types
    =========================================================================
    Demonstrates preprocessing on a custom dataset with:
    - Missing values
    - Categorical variables
    - Numerical variables
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: CUSTOM DATASET WITH MISSING VALUES")
    print("="*80 + "\n")
    
    # -------------------------------------------------------------------------
    # Step 1: Create a synthetic dataset with realistic issues
    # -------------------------------------------------------------------------
    print("[STEP 1] CREATING SYNTHETIC DATASET")
    print("-"*80)
    
    np.random.seed(42)
    n_samples = 5000
    
    # Create dataset with mixed types and missing values
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'], n_samples),
        'experience_years': np.random.randint(0, 40, n_samples),
        'score_1': np.random.uniform(0, 100, n_samples),
        'score_2': np.random.uniform(0, 100, n_samples),
        'score_3': np.random.uniform(0, 100, n_samples),
    }
    
    X = pd.DataFrame(data)
    
    # Create target variable (binary classification)
    y = (X['income'] + X['experience_years'] * 1000 + 
         np.random.normal(0, 10000, n_samples) > 55000).astype(int)
    y = pd.Series(y, name='high_earner')
    
    # Introduce missing values randomly
    for col in ['age', 'income', 'education', 'experience_years']:
        mask = np.random.random(n_samples) < 0.1  # 10% missing
        X.loc[mask, col] = np.nan
    
    # Add some duplicate rows
    duplicate_indices = np.random.choice(X.index, size=20, replace=False)
    X = pd.concat([X, X.loc[duplicate_indices]], ignore_index=True)
    y = pd.concat([y, y.loc[duplicate_indices]], ignore_index=True)
    
    print(f"[OK] Created dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"[INFO] Missing values per column:")
    print(X.isnull().sum())
    print(f"[INFO] Duplicate rows: {X.duplicated().sum()}")
    print(f"\n[INFO] Data types:")
    print(X.dtypes)
    
    # -------------------------------------------------------------------------
    # Step 2: Split data
    # -------------------------------------------------------------------------
    print("\n[STEP 2] SPLITTING DATA")
    print("-"*80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # -------------------------------------------------------------------------
    # Step 3: Configure and apply preprocessing
    # -------------------------------------------------------------------------
    print("\n[STEP 3] PREPROCESSING")
    print("-"*80)
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Configure for handling missing values and categorical data
    config['preprocessing']['remove_duplicates'] = True
    config['preprocessing']['handle_missing'] = True
    config['preprocessing']['missing_numerical_strategy'] = 'median'
    config['preprocessing']['missing_categorical_strategy'] = 'most_frequent'
    config['preprocessing']['encode_categorical'] = True
    config['preprocessing']['categorical_encoding_method'] = 'onehot'
    config['preprocessing']['use_scaler'] = True
    config['preprocessing']['scaling_method'] = 'standard'
    
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_DIR / 'config_custom.yaml', 'w') as f:
        yaml.dump(config, f)
    
    pipeline = PreprocessingPipeline(config_path=str(CONFIG_DIR / 'config_custom.yaml'))
    
    # Fit and transform
    X_train_processed, y_train = pipeline.fit_transform(X_train, y_train)
    X_test_processed = pipeline.transform(X_test)
    
    print(f"\n[INFO] Features after preprocessing: {X_train_processed.shape[1]}")
    print(f"[INFO] Feature names: {pipeline.get_feature_names()[:10]}...")
    
    # -------------------------------------------------------------------------
    # Step 4: Train and evaluate
    # -------------------------------------------------------------------------
    print("\n[STEP 4] MODEL TRAINING")
    print("-"*80)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_processed, y_train)
    
    y_pred = model.predict(X_test_processed)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n[OK] Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low Earner', 'High Earner']))
    
    print("\n" + "="*80)
    print("[OK] EXAMPLE 6 COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")


def main():
    """
    Main function to run all examples.
    """
    print("\n" + "="*80)
    print(" "*20 + "PREPROCESSING PIPELINE EXAMPLES")
    print("="*80)
    print("""
This script demonstrates various use cases of the preprocessing pipeline:

1. Basic Usage - Simple preprocessing with Iris dataset
2. PCA Usage - Dimensionality reduction with Breast Cancer dataset
3. Loading Saved Pipeline - Reusing a previously saved pipeline
4. Feature Selection - Selecting the most important features
5. Complete Workflow - End-to-end ML pipeline
6. Custom Dataset - Handling real-world messy data

Each example is self-contained and can be run independently.
    """)
    
    # Prompt user to select example
    print("="*80)
    print("\nSelect which example(s) to run:")
    print("  1 - Basic Usage")
    print("  2 - PCA Usage")
    print("  3 - Loading Saved Pipeline")
    print("  4 - Feature Selection")
    print("  5 - Complete Workflow")
    print("  6 - Custom Dataset")
    print("  7 - Run ALL examples")
    print("  0 - Exit")
    
    try:
        choice = input("\nEnter your choice (0-7): ").strip()
        
        if choice == '1':
            example_1_basic_usage()
        elif choice == '2':
            example_2_with_pca()
        elif choice == '3':
            example_3_loading_saved_pipeline()
        elif choice == '4':
            example_4_feature_selection()
        elif choice == '5':
            example_5_complete_workflow()
        elif choice == '6':
            example_6_custom_dataset_with_missing_values()
        elif choice == '7':
            print("\n[INFO] Running all examples sequentially...\n")
            example_1_basic_usage()
            example_2_with_pca()
            example_3_loading_saved_pipeline()
            example_4_feature_selection()
            example_5_complete_workflow()
            example_6_custom_dataset_with_missing_values()
            
            print("\n" + "="*80)
            print(" "*25 + "ALL EXAMPLES COMPLETED!")
            print("="*80 + "\n")
        elif choice == '0':
            print("\n[INFO] Exiting...\n")
            return
        else:
            print("\n[WARNING] Invalid choice. Please run again and select 0-7.\n")
    
    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted by user. Exiting...\n")
    except Exception as e:
        print(f"\n[FAILED] An error occurred: {str(e)}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Create directory structure
    for directory in [PIPELINE_DIR, MODEL_DIR, CONFIG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    main()