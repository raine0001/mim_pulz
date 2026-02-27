"""
Utility functions for data processing and feature engineering
in the adaptive AI interpretation project.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


def load_and_preprocess_data(
    file_path: str, target_column: str, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load and preprocess data for machine learning.

    Parameters:
    -----------
    file_path : str
        Path to the data file
    target_column : str
        Name of the target column
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random state for reproducibility

    Returns:
    --------
    X_train, X_test, y_train, y_test : tuple
        Split and preprocessed data
    """
    # Load data
    df = pd.read_csv(file_path)
    logger.info(f"Loaded data with shape: {df.shape}")

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Basic preprocessing
    X = preprocess_features(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic preprocessing to features.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe

    Returns:
    --------
    df_processed : pandas.DataFrame
        Preprocessed dataframe
    """
    df_processed = df.copy()

    # Handle missing values
    for column in df_processed.columns:
        if df_processed[column].dtype in ["int64", "float64"]:
            df_processed[column].fillna(df_processed[column].median(), inplace=True)
        else:
            df_processed[column].fillna(df_processed[column].mode()[0], inplace=True)

    # Encode categorical variables
    categorical_columns = df_processed.select_dtypes(include=["object"]).columns
    for column in categorical_columns:
        le = LabelEncoder()
        df_processed[column] = le.fit_transform(df_processed[column].astype(str))

    return df_processed


def create_adaptive_features(
    df: pd.DataFrame, window_size: int = 5, lag_features: List[str] = None
) -> pd.DataFrame:
    """
    Create adaptive features based on rolling statistics and lags.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    window_size : int
        Size of rolling window for statistical features
    lag_features : list
        List of column names to create lag features for

    Returns:
    --------
    df_enhanced : pandas.DataFrame
        Dataframe with additional adaptive features
    """
    df_enhanced = df.copy()

    # Create rolling statistical features
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        df_enhanced[f"{column}_rolling_mean"] = (
            df[column].rolling(window=window_size).mean()
        )
        df_enhanced[f"{column}_rolling_std"] = (
            df[column].rolling(window=window_size).std()
        )
        df_enhanced[f"{column}_rolling_max"] = (
            df[column].rolling(window=window_size).max()
        )
        df_enhanced[f"{column}_rolling_min"] = (
            df[column].rolling(window=window_size).min()
        )

    # Create lag features if specified
    if lag_features:
        for column in lag_features:
            if column in df.columns:
                for lag in range(1, window_size + 1):
                    df_enhanced[f"{column}_lag_{lag}"] = df[column].shift(lag)

    # Create interaction features
    df_enhanced = create_interaction_features(
        df_enhanced, numeric_columns[:5]
    )  # Limit to avoid explosion

    return df_enhanced


def create_interaction_features(
    df: pd.DataFrame, feature_columns: List[str], interaction_degree: int = 2
) -> pd.DataFrame:
    """
    Create polynomial interaction features.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    feature_columns : list
        List of columns to create interactions for
    interaction_degree : int
        Degree of polynomial interactions

    Returns:
    --------
    df_interactions : pandas.DataFrame
        Dataframe with interaction features added
    """
    df_interactions = df.copy()

    if interaction_degree == 2:
        # Create pairwise interactions
        for i, col1 in enumerate(feature_columns):
            for j, col2 in enumerate(feature_columns[i + 1 :], i + 1):
                interaction_name = f"{col1}_x_{col2}"
                df_interactions[interaction_name] = df[col1] * df[col2]

    return df_interactions


def detect_concept_drift(
    reference_data: pd.DataFrame, current_data: pd.DataFrame, threshold: float = 0.05
) -> Dict[str, bool]:
    """
    Detect concept drift using statistical tests.

    Parameters:
    -----------
    reference_data : pandas.DataFrame
        Reference dataset
    current_data : pandas.DataFrame
        Current dataset to compare
    threshold : float
        P-value threshold for drift detection

    Returns:
    --------
    drift_results : dict
        Dictionary indicating drift for each feature
    """
    from scipy import stats

    drift_results = {}

    for column in reference_data.columns:
        if column in current_data.columns:
            if reference_data[column].dtype in ["int64", "float64"]:
                # Use Kolmogorov-Smirnov test for numerical features
                statistic, p_value = stats.ks_2samp(
                    reference_data[column].dropna(), current_data[column].dropna()
                )
                drift_results[column] = p_value < threshold
            else:
                # Use Chi-square test for categorical features
                ref_counts = reference_data[column].value_counts()
                cur_counts = current_data[column].value_counts()

                # Align the counts
                all_categories = set(ref_counts.index) | set(cur_counts.index)
                ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
                cur_aligned = [cur_counts.get(cat, 0) for cat in all_categories]

                if sum(ref_aligned) > 0 and sum(cur_aligned) > 0:
                    statistic, p_value = stats.chisquare(cur_aligned, ref_aligned)
                    drift_results[column] = p_value < threshold
                else:
                    drift_results[column] = True  # Consider as drift if no data

    return drift_results


def calculate_feature_importance_stability(
    importance_history: List[Dict[str, float]], window_size: int = 5
) -> Dict[str, float]:
    """
    Calculate stability metrics for feature importance over time.

    Parameters:
    -----------
    importance_history : list
        List of feature importance dictionaries over time
    window_size : int
        Window size for stability calculation

    Returns:
    --------
    stability_metrics : dict
        Stability metrics for each feature
    """
    if len(importance_history) < window_size:
        return {}

    stability_metrics = {}
    recent_history = importance_history[-window_size:]

    # Get all feature names
    all_features = set()
    for importance_dict in recent_history:
        all_features.update(importance_dict.keys())

    for feature in all_features:
        importances = [imp_dict.get(feature, 0) for imp_dict in recent_history]
        stability_metrics[feature] = np.std(importances) / (np.mean(importances) + 1e-8)

    return stability_metrics
