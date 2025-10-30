"""
Professional Data Preprocessing Pipeline
==========================================
A comprehensive, modular, and reusable preprocessing pipeline for machine learning projects.
"""

import pandas as pd
import numpy as np
import json
import yaml
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import (
    VarianceThreshold, 
    SelectKBest, 
    f_classif, 
    f_regression,
    chi2,
    mutual_info_classif,
    mutual_info_regression
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


class PreprocessingPipeline:
    """    
    This class handles all preprocessing steps from data loading to feature selection,
    with full configurability and support for different model types.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            config_path (str): Path to the configuration file (YAML or JSON)
        """
        # [INFO] Loading configuration file
        print(f"[INFO] Initializing preprocessing pipeline...")
        self.config = self._load_config(config_path)
        self.pipeline_steps = []
        self.fitted_transformers = {}
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self.categorical_columns = []
        self.numerical_columns = []
        
        # [OK] Configuration loaded successfully
        print(f"[OK] Configuration loaded successfully from {config_path}")
        self._print_config_summary()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML or JSON file.
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        config_file = Path(config_path)
        
        # Check if file exists
        if not config_file.exists():
            print(f"[FAILED] Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load based on file extension
        try:
            if config_file.suffix in ['.yaml', '.yml']:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
            elif config_file.suffix == '.json':
                with open(config_file, 'r') as f:
                    config = json.load(f)
            else:
                print(f"[FAILED] Unsupported configuration format: {config_file.suffix}")
                raise ValueError(f"Unsupported configuration format: {config_file.suffix}")
            
            return config
        except Exception as e:
            print(f"[FAILED] Error loading configuration: {str(e)}")
            raise
    
    def _print_config_summary(self):
        """Print a summary of the loaded configuration."""
        print("\n" + "="*60)
        print("CONFIGURATION SUMMARY")
        print("="*60)
        
        preproc = self.config.get('preprocessing', {})
        
        print(f"Scaling Method: {preproc.get('scaling_method', 'None')}")
        print(f"Use Scaler: {preproc.get('use_scaler', False)}")
        print(f"Use PCA: {preproc.get('use_pca', False)}")
        if preproc.get('use_pca'):
            print(f"  └─ N Components: {preproc.get('n_components', 'auto')}")
        print(f"Use ICA: {preproc.get('use_ica', False)}")
        if preproc.get('use_ica'):
            print(f"  └─ N Components: {preproc.get('ica_components', 'auto')}")
        print(f"Use LDA: {preproc.get('use_lda', False)}")
        if preproc.get('use_lda'):
            print(f"  └─ N Components: {preproc.get('lda_components', 'auto')}")
        print(f"Encode Categorical: {preproc.get('encode_categorical', True)}")
        print(f"Feature Selection: {preproc.get('feature_selection', False)}")
        if preproc.get('feature_selection'):
            print(f"  └─ Method: {preproc.get('selection_method', 'auto')}")
            print(f"  └─ K Best: {preproc.get('k_best', 10)}")
        print(f"Handle Missing: {preproc.get('handle_missing', True)}")
        print(f"Remove Duplicates: {preproc.get('remove_duplicates', True)}")
        print(f"Handle Outliers: {preproc.get('handle_outliers', False)}")
        print("="*60 + "\n")
    
    def fit_transform(self, 
                     X: pd.DataFrame, 
                     y: Optional[pd.Series] = None,
                     target_column: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit the preprocessing pipeline and transform the data.
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series, optional): Target variable (required for supervised DR like LDA)
            target_column (str, optional): Name of target column if included in X
            
        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: Transformed features and target
        """
        print("\n" + "="*60)
        print("STARTING PREPROCESSING PIPELINE - FIT & TRANSFORM")
        print("="*60 + "\n")
        
        # Make a copy to avoid modifying original data
        X_processed = X.copy()
        
        # Extract target if provided in X
        if target_column and target_column in X_processed.columns:
            y = X_processed[target_column].copy()
            X_processed = X_processed.drop(columns=[target_column])
            print(f"[INFO] Extracted target column: {target_column}")
        
        # Store original feature names
        self.feature_names_in_ = X_processed.columns.tolist()
        print(f"[INFO] Input features: {len(self.feature_names_in_)} columns")
        
        # =====================================================================
        # STEP 1: DATA CLEANING
        # =====================================================================
        print("\n" + "-"*60)
        print("[STEP 1] DATA CLEANING")
        print("-"*60)
        
        X_processed, y = self._step1_data_cleaning(X_processed, y)
        
        # =====================================================================
        # STEP 2: MISSING VALUES HANDLING
        # =====================================================================
        print("\n" + "-"*60)
        print("[STEP 2] MISSING VALUES HANDLING")
        print("-"*60)
        
        X_processed = self._step2_handle_missing(X_processed)
        
        # =====================================================================
        # STEP 3: IDENTIFY COLUMN TYPES
        # =====================================================================
        print("\n" + "-"*60)
        print("[STEP 3] COLUMN TYPE IDENTIFICATION")
        print("-"*60)
        
        self._identify_column_types(X_processed)
        
        # =====================================================================
        # STEP 4: CATEGORICAL ENCODING
        # =====================================================================
        print("\n" + "-"*60)
        print("[STEP 4] CATEGORICAL ENCODING")
        print("-"*60)
        
        X_processed = self._step4_encode_categorical(X_processed)
        
        # =====================================================================
        # STEP 5: STATISTICAL FILTERING
        # =====================================================================
        print("\n" + "-"*60)
        print("[STEP 5] STATISTICAL FILTERING")
        print("-"*60)
        
        X_processed = self._step5_statistical_filtering(X_processed)
        
        # =====================================================================
        # STEP 6: SCALING/NORMALIZATION
        # =====================================================================
        print("\n" + "-"*60)
        print("[STEP 6] SCALING/NORMALIZATION")
        print("-"*60)
        
        X_processed = self._step6_scaling(X_processed)
        
        # =====================================================================
        # STEP 7: DIMENSIONALITY REDUCTION
        # =====================================================================
        print("\n" + "-"*60)
        print("[STEP 7] DIMENSIONALITY REDUCTION")
        print("-"*60)
        
        X_processed = self._step7_dimensionality_reduction(X_processed, y)
        
        # =====================================================================
        # STEP 8: FEATURE SELECTION
        # =====================================================================
        print("\n" + "-"*60)
        print("[STEP 8] FEATURE SELECTION")
        print("-"*60)
        
        X_processed = self._step8_feature_selection(X_processed, y)
        
        # Store final feature names
        if isinstance(X_processed, pd.DataFrame):
            self.feature_names_out_ = X_processed.columns.tolist()
        else:
            self.feature_names_out_ = [f"feature_{i}" for i in range(X_processed.shape[1])]
        
        print("\n" + "="*60)
        print("[OK] PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"[INFO] Final shape: {X_processed.shape}")
        print(f"[INFO] Input features: {len(self.feature_names_in_)}")
        print(f"[INFO] Output features: {len(self.feature_names_out_)}")
        print("="*60 + "\n")
        
        # Convert to numpy array for consistency
        if isinstance(X_processed, pd.DataFrame):
            X_processed = X_processed.values
        
        return X_processed, y
    
    def _step1_data_cleaning(self, X: pd.DataFrame, y: Optional[pd.Series]) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Step 1: Clean the data by removing duplicates and handling outliers.
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series, optional): Target variable
            
        Returns:
            Tuple[pd.DataFrame, Optional[pd.Series]]: Cleaned features and target
        """
        config = self.config.get('preprocessing', {})
        initial_rows = len(X)
        
        # Sub-step 1.1: Remove duplicates
        if config.get('remove_duplicates', True):
            print("[INFO] Checking for duplicate rows...")
            duplicates = X.duplicated().sum()
            
            if duplicates > 0:
                print(f"[WARNING] Found {duplicates} duplicate rows")
                X = X.drop_duplicates()
                
                # Align target variable if provided
                if y is not None:
                    y = y[X.index]
                
                print(f"[OK] Removed {duplicates} duplicate rows")
            else:
                print("[OK] No duplicate rows found")
        
        # Sub-step 1.2: Handle outliers (optional)
        if config.get('handle_outliers', False):
            print("[INFO] Detecting and handling outliers...")
            outlier_method = config.get('outlier_method', 'iqr')
            
            if outlier_method == 'iqr':
                # IQR method for outlier detection
                X, y = self._remove_outliers_iqr(X, y)
            elif outlier_method == 'zscore':
                # Z-score method for outlier detection
                X, y = self._remove_outliers_zscore(X, y)
            else:
                print(f"[WARNING] Unknown outlier method: {outlier_method}")
        
        rows_removed = initial_rows - len(X)
        if rows_removed > 0:
            print(f"[OK] Data cleaning complete. Removed {rows_removed} rows ({rows_removed/initial_rows*100:.2f}%)")
        else:
            print("[OK] Data cleaning complete. No rows removed")
        
        return X, y
    
    def _remove_outliers_iqr(self, X: pd.DataFrame, y: Optional[pd.Series]) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Remove outliers using the IQR (Interquartile Range) method.
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series, optional): Target variable
            
        Returns:
            Tuple[pd.DataFrame, Optional[pd.Series]]: Data without outliers
        """
        # Only apply to numerical columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            print("[INFO] No numerical columns for outlier detection")
            return X, y
        
        # Calculate IQR for each numerical column
        Q1 = X[numerical_cols].quantile(0.25)
        Q3 = X[numerical_cols].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier boundaries
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Create mask for rows without outliers
        mask = ~((X[numerical_cols] < lower_bound) | (X[numerical_cols] > upper_bound)).any(axis=1)
        
        outliers_removed = (~mask).sum()
        
        if outliers_removed > 0:
            X = X[mask]
            if y is not None:
                y = y[X.index]
            print(f"[OK] Removed {outliers_removed} outlier rows using IQR method")
        else:
            print("[OK] No outliers detected using IQR method")
        
        return X, y
    
    def _remove_outliers_zscore(self, X: pd.DataFrame, y: Optional[pd.Series], threshold: float = 3.0) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Remove outliers using the Z-score method.
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series, optional): Target variable
            threshold (float): Z-score threshold for outlier detection
            
        Returns:
            Tuple[pd.DataFrame, Optional[pd.Series]]: Data without outliers
        """
        # Only apply to numerical columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            print("[INFO] No numerical columns for outlier detection")
            return X, y
        
        # Calculate Z-scores
        z_scores = np.abs((X[numerical_cols] - X[numerical_cols].mean()) / X[numerical_cols].std())
        
        # Create mask for rows without outliers
        mask = (z_scores < threshold).all(axis=1)
        
        outliers_removed = (~mask).sum()
        
        if outliers_removed > 0:
            X = X[mask]
            if y is not None:
                y = y[X.index]
            print(f"[OK] Removed {outliers_removed} outlier rows using Z-score method (threshold={threshold})")
        else:
            print("[OK] No outliers detected using Z-score method")
        
        return X, y
    
    def _step2_handle_missing(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2: Handle missing values in the dataset.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Features with missing values handled
        """
        config = self.config.get('preprocessing', {})
        
        if not config.get('handle_missing', True):
            print("[INFO] Missing value handling is disabled in configuration")
            return X
        
        # Check for missing values
        missing_counts = X.isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing == 0:
            print("[OK] No missing values detected")
            return X
        
        print(f"[WARNING] Found {total_missing} missing values across {(missing_counts > 0).sum()} columns")
        
        # Separate numerical and categorical columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Handle numerical missing values
        if len(numerical_cols) > 0:
            numerical_strategy = config.get('missing_numerical_strategy', 'mean')
            print(f"[INFO] Imputing numerical columns with strategy: {numerical_strategy}")
            
            imputer = SimpleImputer(strategy=numerical_strategy)
            X[numerical_cols] = imputer.fit_transform(X[numerical_cols])
            
            # Store the imputer for later use
            self.fitted_transformers['numerical_imputer'] = imputer
            print(f"[OK] Imputed {len(numerical_cols)} numerical columns")
        
        # Handle categorical missing values
        if len(categorical_cols) > 0:
            categorical_strategy = config.get('missing_categorical_strategy', 'most_frequent')
            print(f"[INFO] Imputing categorical columns with strategy: {categorical_strategy}")
            
            imputer = SimpleImputer(strategy=categorical_strategy)
            X[categorical_cols] = imputer.fit_transform(X[categorical_cols])
            
            # Store the imputer for later use
            self.fitted_transformers['categorical_imputer'] = imputer
            print(f"[OK] Imputed {len(categorical_cols)} categorical columns")
        
        # Verify no missing values remain
        remaining_missing = X.isnull().sum().sum()
        if remaining_missing == 0:
            print("[OK] All missing values handled successfully")
        else:
            print(f"[WARNING] {remaining_missing} missing values still remain")
        
        return X
    
    def _identify_column_types(self, X: pd.DataFrame):
        """
        Step 3: Identify numerical and categorical columns.
        
        Args:
            X (pd.DataFrame): Input features
        """
        # Identify numerical columns
        self.numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Identify categorical columns
        self.categorical_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        print(f"[INFO] Identified {len(self.numerical_columns)} numerical columns")
        print(f"[INFO] Identified {len(self.categorical_columns)} categorical columns")
        
        if len(self.categorical_columns) > 0:
            print(f"[INFO] Categorical columns: {', '.join(self.categorical_columns[:5])}" + 
                  (f" ... and {len(self.categorical_columns) - 5} more" if len(self.categorical_columns) > 5 else ""))
    
    def _step4_encode_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Step 4: Encode categorical variables.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Features with encoded categorical variables
        """
        config = self.config.get('preprocessing', {})
        
        if not config.get('encode_categorical', True):
            print("[INFO] Categorical encoding is disabled in configuration")
            return X
        
        if len(self.categorical_columns) == 0:
            print("[INFO] No categorical columns to encode")
            return X
        
        print(f"[INFO] Encoding {len(self.categorical_columns)} categorical columns...")
        
        encoding_method = config.get('categorical_encoding_method', 'onehot')
        
        if encoding_method == 'onehot':
            # One-Hot Encoding for nominal categorical variables
            print("[INFO] Using One-Hot Encoding")
            
            # Apply one-hot encoding
            X_encoded = pd.get_dummies(X, columns=self.categorical_columns, drop_first=False)
            
            # Store the mapping for later use
            self.fitted_transformers['categorical_columns'] = self.categorical_columns
            self.fitted_transformers['encoding_method'] = 'onehot'
            
            print(f"[OK] One-Hot Encoding complete. Features increased from {X.shape[1]} to {X_encoded.shape[1]}")
            return X_encoded
            
        elif encoding_method == 'label':
            # Label Encoding for ordinal categorical variables
            print("[INFO] Using Label Encoding")
            
            label_encoders = {}
            for col in self.categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
            
            # Store the encoders for later use
            self.fitted_transformers['label_encoders'] = label_encoders
            self.fitted_transformers['encoding_method'] = 'label'
            
            print(f"[OK] Label Encoding complete for {len(self.categorical_columns)} columns")
            return X
        
        else:
            print(f"[WARNING] Unknown encoding method: {encoding_method}. Skipping encoding.")
            return X
    
    def _step5_statistical_filtering(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Step 5: Apply statistical filtering (low variance and high correlation).
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Filtered features
        """
        config = self.config.get('preprocessing', {})
        
        # Sub-step 5.1: Remove low variance features
        if config.get('remove_low_variance', True):
            print("[INFO] Removing low variance features...")
            
            variance_threshold = config.get('variance_threshold', 0.01)
            initial_features = X.shape[1]
            
            # Apply variance threshold
            selector = VarianceThreshold(threshold=variance_threshold)
            X_filtered = selector.fit_transform(X)
            
            # Get selected feature names
            if isinstance(X, pd.DataFrame):
                selected_features = X.columns[selector.get_support()].tolist()
                X = pd.DataFrame(X_filtered, columns=selected_features, index=X.index)
            else:
                X = X_filtered
            
            # Store the selector
            self.fitted_transformers['variance_selector'] = selector
            
            features_removed = initial_features - X.shape[1]
            print(f"[OK] Removed {features_removed} low variance features (threshold={variance_threshold})")
        
        # Sub-step 5.2: Remove highly correlated features
        if config.get('remove_correlated', True):
            print("[INFO] Removing highly correlated features...")
            
            correlation_threshold = config.get('correlation_threshold', 0.95)
            initial_features = X.shape[1]
            
            # Calculate correlation matrix
            if isinstance(X, pd.DataFrame):
                corr_matrix = X.corr().abs()
            else:
                corr_matrix = pd.DataFrame(X).corr().abs()
            
            # Select upper triangle of correlation matrix
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            # Find features with correlation greater than threshold
            to_drop = [column for column in upper_triangle.columns 
                    if any(upper_triangle[column] > correlation_threshold)]

            if len(to_drop) > 0:
                X = X.drop(columns=to_drop) if isinstance(X, pd.DataFrame) else np.delete(X, to_drop, axis=1)
                print(f"[OK] Removed {len(to_drop)} highly correlated features (threshold={correlation_threshold})")
                
                # Store the KEPT columns indices/names 
                if isinstance(X, pd.DataFrame):
                    self.fitted_transformers['kept_features_after_correlation'] = X.columns.tolist()
                else:
                    kept_indices = [i for i in range(len(upper_triangle.columns)) if upper_triangle.columns[i] not in to_drop]
                    self.fitted_transformers['kept_features_indices'] = kept_indices
        return X
    
    def _step6_scaling(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Step 6: Scale/normalize the features.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Scaled features
        """
        config = self.config.get('preprocessing', {})
        
        if not config.get('use_scaler', True):
            print("[INFO] Scaling is disabled in configuration")
            return X
        
        scaling_method = config.get('scaling_method', 'standard')
        print(f"[INFO] Applying {scaling_method} scaling...")
        
        # Convert to numpy for scaling
        feature_names = X.columns if isinstance(X, pd.DataFrame) else None
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        
        # Select scaler based on configuration
        if scaling_method == 'standard':
            # Standardization: (x - mean) / std
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            # Min-Max normalization: (x - min) / (max - min)
            scaler = MinMaxScaler()
        else:
            print(f"[WARNING] Unknown scaling method: {scaling_method}. Using StandardScaler.")
            scaler = StandardScaler()
        
        # Fit and transform
        X_scaled = scaler.fit_transform(X_values)
        
        # Store the scaler
        self.fitted_transformers['scaler'] = scaler
        
        # Convert back to DataFrame if needed
        if feature_names is not None:
            X_scaled = pd.DataFrame(X_scaled, columns=feature_names, index=X.index)
        
        print(f"[OK] {scaling_method.capitalize()} scaling applied successfully")
        
        return X_scaled
    
    def _step7_dimensionality_reduction(self, X: pd.DataFrame, y: Optional[pd.Series]) -> pd.DataFrame:
        """
        Step 7: Apply dimensionality reduction techniques.
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series, optional): Target variable (required for LDA)
            
        Returns:
            pd.DataFrame: Reduced features
        """
        config = self.config.get('preprocessing', {})
        
        # Convert to numpy array
        X_values = X.values if isinstance(X, pd.DataFrame) else X

        # Sub-step 7.1: PCA (Principal Component Analysis)
        if config.get('use_pca', False):
            print("[INFO] Applying PCA (Principal Component Analysis)...")
            
            n_components = config.get('n_components', None)
            
            # If n_components is 'auto' or None, use min(n_samples, n_features) - 1
            if n_components == 'auto' or n_components is None:
                n_components = min(X_values.shape[0], X_values.shape[1]) - 1
                print(f"[INFO] Auto-selecting {n_components} components")
            else:
                # Ensure n_components doesn't exceed available features
                max_components = min(X_values.shape[0], X_values.shape[1])
                if n_components > max_components:
                    print(f"[WARNING] Requested {n_components} components but only {max_components} available")
                    n_components = max_components
                    print(f"[INFO] Adjusted to {n_components} components")
            
            pca = PCA(n_components=n_components)
            X_values = pca.fit_transform(X_values)

        # Sub-step 7.2: ICA (Independent Component Analysis)
        elif config.get('use_ica', False):
            print("[INFO] Applying ICA (Independent Component Analysis)...")
            
            n_components = config.get('ica_components', None)
            
            # If n_components is 'auto' or None, use min(n_samples, n_features)
            if n_components == 'auto' or n_components is None:
                n_components = min(X_values.shape[0], X_values.shape[1])
                print(f"[INFO] Auto-selecting {n_components} components")
            else:
                # Ensure n_components doesn't exceed available features
                max_components = min(X_values.shape[0], X_values.shape[1])
                if n_components > max_components:
                    print(f"[WARNING] Requested {n_components} components but only {max_components} available")
                    n_components = max_components
                    print(f"[INFO] Adjusted to {n_components} components")
            
            ica = FastICA(n_components=n_components, random_state=42)
            X_values = ica.fit_transform(X_values)
        
        # Sub-step 7.3: LDA (Linear Discriminant Analysis) - Supervised
        elif config.get('use_lda', False):
            if y is None:
                print("[WARNING] LDA requires target variable (y). Skipping LDA.")
            else:
                print("[INFO] Applying LDA (Linear Discriminant Analysis)...")
                
                n_components = config.get('lda_components', None)
                
                # LDA components limited by min(n_features, n_classes - 1)
                n_classes = len(np.unique(y))
                max_components = min(X_values.shape[1], n_classes - 1)
                
                if n_components is None or n_components == 'auto':
                    n_components = max_components
                    print(f"[INFO] Auto-selecting {n_components} components")
                else:
                    n_components = min(n_components, max_components)
                
                lda = LDA(n_components=n_components)
                X_values = lda.fit_transform(X_values, y)
                
                # Store LDA transformer
                self.fitted_transformers['lda'] = lda
                
                # Calculate explained variance
                explained_variance = lda.explained_variance_ratio_.sum() * 100
                print(f"[OK] LDA complete. Reduced to {n_components} discriminant components")
                print(f"[INFO] Explained variance: {explained_variance:.2f}%")
                
                # Update feature names
                X = pd.DataFrame(X_values, columns=[f'LD{i+1}' for i in range(n_components)])
        else:
            print("[INFO] No dimensionality reduction method selected")
        
        return X
    
    def _step8_feature_selection(self, X: pd.DataFrame, y: Optional[pd.Series]) -> pd.DataFrame:
        """
        Step 8: Apply feature selection to keep only the most relevant features.
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series, optional): Target variable
            
        Returns:
            pd.DataFrame: Selected features
        """
        config = self.config.get('preprocessing', {})
        
        if not config.get('feature_selection', False):
            print("[INFO] Feature selection is disabled in configuration")
            return X
        
        if y is None:
            print("[WARNING] Feature selection requires target variable (y). Skipping feature selection.")
            return X
        
        selection_method = config.get('selection_method', 'auto')
        k_best = config.get('k_best', 10)
        
        print(f"[INFO] Applying feature selection (method: {selection_method})...")
        
        # Convert to numpy array
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        feature_names = X.columns if isinstance(X, pd.DataFrame) else [f'feature_{i}' for i in range(X.shape[1])]
        
        # Limit k_best to available features
        k_best = min(k_best, X_values.shape[1])
        
        # Determine if problem is classification or regression
        problem_type = config.get('problem_type', 'auto')
        
        if problem_type == 'auto':
            # Auto-detect: if target has few unique values, assume classification
            n_unique = len(np.unique(y))
            if n_unique < 20:
                problem_type = 'classification'
            else:
                problem_type = 'regression'
            print(f"[INFO] Auto-detected problem type: {problem_type}")
        
        # Select appropriate scoring function
        if selection_method == 'auto':
            if problem_type == 'classification':
                selector = SelectKBest(score_func=f_classif, k=k_best)
            else:
                selector = SelectKBest(score_func=f_regression, k=k_best)
        
        elif selection_method == 'f_test':
            if problem_type == 'classification':
                selector = SelectKBest(score_func=f_classif, k=k_best)
            else:
                selector = SelectKBest(score_func=f_regression, k=k_best)
        
        elif selection_method == 'mutual_info':
            if problem_type == 'classification':
                selector = SelectKBest(score_func=mutual_info_classif, k=k_best)
            else:
                selector = SelectKBest(score_func=mutual_info_regression, k=k_best)
        
        elif selection_method == 'chi2':
            if problem_type == 'classification':
                # Chi2 requires non-negative features
                if np.any(X_values < 0):
                    print("[WARNING] Chi2 requires non-negative features. Switching to f_classif.")
                    selector = SelectKBest(score_func=f_classif, k=k_best)
                else:
                    selector = SelectKBest(score_func=chi2, k=k_best)
            else:
                print("[WARNING] Chi2 is only for classification. Using f_regression instead.")
                selector = SelectKBest(score_func=f_regression, k=k_best)
        
        elif selection_method == 'random_forest':
            print("[INFO] Using Random Forest feature importance...")
            
            # Use Random Forest to determine feature importance
            if problem_type == 'classification':
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            rf.fit(X_values, y)
            
            # Get feature importances
            importances = rf.feature_importances_
            
            # Select top k features
            indices = np.argsort(importances)[::-1][:k_best]
            
            X_selected = X_values[:, indices]
            selected_features = [feature_names[i] for i in indices]
            
            # Store the selected features and model
            self.fitted_transformers['feature_selector'] = {'method': 'random_forest', 'indices': indices}
            
            print(f"[OK] Random Forest feature selection complete. Selected {k_best} features")
            
            return pd.DataFrame(X_selected, columns=selected_features)
        
        else:
            print(f"[WARNING] Unknown selection method: {selection_method}. Skipping feature selection.")
            return X
        
        # Fit and transform (for non-RF methods)
        X_selected = selector.fit_transform(X_values, y)
        
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_features = [feature_names[i] for i in selected_indices]
        
        # Store the selector
        self.fitted_transformers['feature_selector'] = selector
        
        print(f"[OK] Feature selection complete. Selected {k_best} out of {X_values.shape[1]} features")
        
        return pd.DataFrame(X_selected, columns=selected_features)
    
    def transform(self, X: pd.DataFrame, target_column: Optional[str] = None) -> np.ndarray:
        """
        Transform new data using the fitted pipeline.
        
        Args:
            X (pd.DataFrame): Input features to transform
            target_column (str, optional): Name of target column if included in X
            
        Returns:
            np.ndarray: Transformed features
        """
        print("\n" + "="*60)
        print("APPLYING PREPROCESSING PIPELINE - TRANSFORM ONLY")
        print("="*60 + "\n")
        
        # Make a copy to avoid modifying original data
        X_processed = X.copy()
        
        # Remove target if provided
        if target_column and target_column in X_processed.columns:
            X_processed = X_processed.drop(columns=[target_column])
            print(f"[INFO] Removed target column: {target_column}")
        
        # Ensure input features match training features
        if self.feature_names_in_ is not None:
            missing_features = set(self.feature_names_in_) - set(X_processed.columns)
            extra_features = set(X_processed.columns) - set(self.feature_names_in_)
            
            if missing_features:
                print(f"[WARNING] Missing features: {missing_features}")
            if extra_features:
                print(f"[WARNING] Extra features (will be ignored): {extra_features}")
            
            # Reorder and select only training features
            X_processed = X_processed[self.feature_names_in_]
        
        config = self.config.get('preprocessing', {})
        
        # Apply missing value imputation
        if config.get('handle_missing', True):
            if 'numerical_imputer' in self.fitted_transformers:
                numerical_cols = X_processed.select_dtypes(include=[np.number]).columns.tolist()
                X_processed[numerical_cols] = self.fitted_transformers['numerical_imputer'].transform(X_processed[numerical_cols])
            
            if 'categorical_imputer' in self.fitted_transformers:
                categorical_cols = X_processed.select_dtypes(exclude=[np.number]).columns.tolist()
                X_processed[categorical_cols] = self.fitted_transformers['categorical_imputer'].transform(X_processed[categorical_cols])
        
        # Apply categorical encoding
        if config.get('encode_categorical', True):
            if 'encoding_method' in self.fitted_transformers:
                if self.fitted_transformers['encoding_method'] == 'onehot':
                    # Apply same one-hot encoding as training
                    categorical_cols = self.fitted_transformers.get('categorical_columns', [])
                    X_processed = pd.get_dummies(X_processed, columns=categorical_cols, drop_first=False)
                
                elif self.fitted_transformers['encoding_method'] == 'label':
                    # Apply same label encoding as training
                    label_encoders = self.fitted_transformers.get('label_encoders', {})
                    for col, le in label_encoders.items():
                        if col in X_processed.columns:
                            X_processed[col] = le.transform(X_processed[col].astype(str))
        
        # Apply variance filtering
        if 'variance_selector' in self.fitted_transformers:
            X_processed = self.fitted_transformers['variance_selector'].transform(X_processed)

        # Remove correlated features
        if 'kept_features_after_correlation' in self.fitted_transformers:
            kept_features = self.fitted_transformers['kept_features_after_correlation']
            if isinstance(X_processed, pd.DataFrame):
                X_processed = X_processed[kept_features]
            else:
                # X_processed est déjà un array après variance_selector
                # Il faut trouver les indices par rapport aux features APRÈS variance filtering
                if 'variance_selector' in self.fitted_transformers:
                    # Get features after variance filtering
                    variance_selector = self.fitted_transformers['variance_selector']
                    features_after_variance = [self.feature_names_in_[i] for i in range(len(self.feature_names_in_)) 
                                            if variance_selector.get_support()[i]]
                else:
                    features_after_variance = self.feature_names_in_
                
                # Find indices of kept features relative to features_after_variance
                kept_indices = [features_after_variance.index(f) for f in kept_features if f in features_after_variance]
                X_processed = X_processed[:, kept_indices]
        
        # Apply scaling
        if 'scaler' in self.fitted_transformers:
            X_processed = self.fitted_transformers['scaler'].transform(X_processed)
        
        # Apply dimensionality reduction
        if 'pca' in self.fitted_transformers:
            X_processed = self.fitted_transformers['pca'].transform(X_processed)
        elif 'ica' in self.fitted_transformers:
            X_processed = self.fitted_transformers['ica'].transform(X_processed)
        elif 'lda' in self.fitted_transformers:
            X_processed = self.fitted_transformers['lda'].transform(X_processed)
        
        # Apply feature selection
        if 'feature_selector' in self.fitted_transformers:
            selector = self.fitted_transformers['feature_selector']
            if isinstance(selector, dict) and selector.get('method') == 'random_forest':
                # Random Forest method: use stored indices
                X_processed = X_processed[:, selector['indices']]
            else:
                # Statistical methods: use transform
                X_processed = selector.transform(X_processed)
        
        print("[OK] Transform complete")
        print(f"[INFO] Output shape: {X_processed.shape if hasattr(X_processed, 'shape') else len(X_processed)}")
        print("="*60 + "\n")
        
        return X_processed
    
    def save_pipeline(self, filepath: str):
        """
        Save the fitted pipeline to disk.
        
        Args:
            filepath (str): Path to save the pipeline
        """
        print(f"[INFO] Saving pipeline to {filepath}...")
        
        # Create pipeline object with all necessary components
        pipeline_data = {
            'config': self.config,
            'fitted_transformers': self.fitted_transformers,
            'feature_names_in_': self.feature_names_in_,
            'feature_names_out_': self.feature_names_out_,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save using joblib
        try:
            joblib.dump(pipeline_data, filepath)
            print(f"[OK] Pipeline saved successfully to {filepath}")
        except Exception as e:
            print(f"[FAILED] Error saving pipeline: {str(e)}")
            raise
    
    @classmethod
    def load_pipeline(cls, filepath: str) -> 'PreprocessingPipeline':
        """
        Load a fitted pipeline from disk.
        
        Args:
            filepath (str): Path to the saved pipeline
            
        Returns:
            PreprocessingPipeline: Loaded pipeline instance
        """
        print(f"[INFO] Loading pipeline from {filepath}...")
        
        try:
            # Load pipeline data
            pipeline_data = joblib.load(filepath)
            
            # Create a temporary config file (in-memory)
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(pipeline_data['config'], f)
                temp_config_path = f.name
            
            # Create new pipeline instance
            pipeline = cls(temp_config_path)
            
            # Restore fitted components
            pipeline.fitted_transformers = pipeline_data['fitted_transformers']
            pipeline.feature_names_in_ = pipeline_data['feature_names_in_']
            pipeline.feature_names_out_ = pipeline_data['feature_names_out_']
            pipeline.categorical_columns = pipeline_data['categorical_columns']
            pipeline.numerical_columns = pipeline_data['numerical_columns']
            
            # Clean up temp file
            import os
            os.unlink(temp_config_path)
            
            print(f"[OK] Pipeline loaded successfully from {filepath}")
            print(f"[INFO] Pipeline was saved on: {pipeline_data.get('timestamp', 'Unknown')}")
            
            return pipeline
            
        except Exception as e:
            print(f"[FAILED] Error loading pipeline: {str(e)}")
            raise
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of output features after preprocessing.
        
        Returns:
            List[str]: List of feature names
        """
        return self.feature_names_out_ if self.feature_names_out_ is not None else []
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return self.config
    
    def print_summary(self):
        """Print a summary of the preprocessing pipeline."""
        print("\n" + "="*60)
        print("PREPROCESSING PIPELINE SUMMARY")
        print("="*60)
        
        print(f"\nInput Features: {len(self.feature_names_in_) if self.feature_names_in_ else 0}")
        print(f"Output Features: {len(self.feature_names_out_) if self.feature_names_out_ else 0}")
        
        print(f"\nFitted Transformers:")
        for name, transformer in self.fitted_transformers.items():
            print(f"  - {name}: {type(transformer).__name__}")
        
        print("\n" + "="*60 + "\n")

    def save_pipeline(self, filepath: str):
        """
        Save the fitted pipeline to disk.
        
        Args:
            filepath (str or Path): Path to save the pipeline
        """
        filepath = Path(filepath)
        
        # Create parent directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"[INFO] Saving pipeline to {filepath}...")
        
        # Create pipeline object with all necessary components
        pipeline_data = {
            'config': self.config,
            'fitted_transformers': self.fitted_transformers,
            'feature_names_in_': self.feature_names_in_,
            'feature_names_out_': self.feature_names_out_,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save using joblib
        try:
            joblib.dump(pipeline_data, str(filepath))
            print(f"[OK] Pipeline saved successfully to {filepath}")
        except Exception as e:
            print(f"[FAILED] Error saving pipeline: {str(e)}")
            raise