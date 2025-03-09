# %load submissions/starting_kit/estimator.py

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_is_fitted
from sklearn import set_config

set_config(transform_output="pandas")


###############################################################################
# Step 1: Data Filtering and Preprocessing Transformers
###############################################################################

class DomainFilter(BaseEstimator, TransformerMixin):
    """
    Keeps only the rows matching the target domain.
    """
    def __init__(self, target_domain='v'):
        self.target_domain = target_domain

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return X[X['domain'] == self.target_domain]


class DataFrameCleanerCatBoost(BaseEstimator, TransformerMixin):
    """
    Performs cleaning and feature engineering.
      - Leaves 'gender' unchanged so that CatBoost can treat it as a categorical feature.
      - Drops 'domain' once filtered.
      - Computes aggregate features for ECG and PPG signals.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Drop the domain column (already used for filtering)
        X = X.drop(['domain'], axis=1)
        # Compute aggregate features for ECG and PPG signals
        X['ecg_mean'] = X['ecg'].apply(lambda x: x.mean())
        X['ppg_mean'] = X['ppg'].apply(lambda x: x.mean())
        X['var_ecg'] = X['ecg'].apply(lambda x: x.var())
        X['var_ppg'] = X['ppg'].apply(lambda x: x.var())
        # Drop the raw waveform columns as they are no longer needed
        X = X.drop(['ecg', 'ppg'], axis=1)
        return X


###############################################################################
# Step 2: CatBoost Wrapper that Ignores Invalid Targets
###############################################################################

class IgnoreDomainCatBoost(BaseEstimator, RegressorMixin):
    """
    A wrapper around CatBoostRegressor that:
      - Removes rows where y == -1 (missing targets).
      - Trains CatBoost with provided hyperparameters and specifies the categorical feature indices.
    """
    def __init__(self, catboost_params=None, cat_features=None):
        """
        catboost_params: dict of parameters for CatBoostRegressor.
        cat_features: list of column indices (in the transformed array) that are categorical.
        """
        self.catboost_params = catboost_params or {}
        self.cat_features = cat_features
        self.model_ = None

    def fit(self, X, y):
        # Filter out invalid targets
        mask = (y != -1)
        X = X[mask]
        y = y[mask]
        # Initialize CatBoostRegressor with specified parameters
        self.model_ = CatBoostRegressor(**self.catboost_params)
        self.model_.fit(X, y, cat_features=self.cat_features, verbose=False)
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict(X)


###############################################################################
# Step 3: Build the Pipeline
###############################################################################

def get_estimator():
    """
    Returns a pipeline that:
      1) Filters the data to the target 'v' domain.
      2) Performs cleaning and feature engineering.
      3) Applies a ColumnTransformer:
           - Imputing numeric features.
           - Passing categorical features ("gender") unchanged.
      4) Trains a CatBoost model that drops samples with y == -1.
    """
    
    # Define features after cleaning:
    # Numeric features: age, and our computed aggregate features.
    numeric_features = ["age", "ecg_mean", "ppg_mean", "var_ecg", "var_ppg"]
    # Categorical feature: keep 'gender' as is so that CatBoost handles it.
    categorical_features = ["gender"]

    # Numeric transformer: here only imputation is applied.
    # (CatBoost is insensitive to scaling so we skip scaling.)
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean"))
    ])

    # Combine numeric and categorical pipelines.
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", "passthrough", categorical_features),
    ])

    # After the ColumnTransformer, the numeric features come first
    # followed by categorical features.
    # For our case, categorical columns start at position len(numeric_features)
    cat_feature_indices = list(range(len(numeric_features), len(numeric_features) + len(categorical_features)))

    # CatBoost hyperparameters (adjust as needed)
    catboost_params = {
        "iterations": 200,
        "learning_rate": 0.1,
        "depth": 6,
        "random_seed": 42,
    }

    pipeline = Pipeline([
        # ("domain_filter", DomainFilter(target_domain='v')),
        ("cleaner", DataFrameCleanerCatBoost()),
        ("preprocessor", preprocessor),
        ("catboost", IgnoreDomainCatBoost(
            catboost_params=catboost_params,
            cat_features=cat_feature_indices
        ))
    ])

    return pipeline
