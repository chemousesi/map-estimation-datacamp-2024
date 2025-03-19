from catboost import CatBoostRegressor
from sklearn.base import (
    BaseEstimator,
    RegressorMixin,
    TransformerMixin,
)
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
    def __init__(self, target_domain='v'):
        self.target_domain = target_domain

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return X[X['domain'] == self.target_domain]


class DataFrameCleanerCatBoost(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(['domain'], axis=1)
        X['ecg_mean'] = X['ecg'].apply(lambda x: x.mean())
        X['ppg_mean'] = X['ppg'].apply(lambda x: x.mean())
        X['var_ecg'] = X['ecg'].apply(lambda x: x.var())
        X['var_ppg'] = X['ppg'].apply(lambda x: x.var())
        X = X.drop(['ecg', 'ppg'], axis=1)
        return X


###############################################################################
# Step 2: CatBoost Wrapper that Ignores Invalid Targets
###############################################################################


class IgnoreDomainCatBoost(BaseEstimator, RegressorMixin):
    def __init__(self, catboost_params=None, cat_features=None):
        self.catboost_params = catboost_params or {}
        self.cat_features = cat_features
        self.model_ = None

    def fit(self, X, y):
        mask = y != -1
        X = X[mask]
        y = y[mask]
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
    numeric_features = ["age", "ecg_mean", "ppg_mean", "var_ecg", "var_ppg"]
    categorical_features = ["gender"]

    numeric_transformer = Pipeline(
        [("imputer", SimpleImputer(strategy="mean"))]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_features),
            ("cat", "passthrough", categorical_features),
        ]
    )

    cat_feature_indices = list(
        range(len(numeric_features), len(numeric_features) + len(categorical_features))
    )

    catboost_params = {
        "iterations": 200,
        "learning_rate": 0.1,
        "depth": 6,
        "random_seed": 42,
    }

    pipeline = Pipeline(
        [
            ("cleaner", DataFrameCleanerCatBoost()),
            ("preprocessor", preprocessor),
            (
                "catboost",
                IgnoreDomainCatBoost(
                    catboost_params=catboost_params,
                    cat_features=cat_feature_indices,
                ),
            ),
        ]
    )

    return pipeline
