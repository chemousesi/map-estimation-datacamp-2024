# %load submissions/starting_kit/estimator.py

from sklearn import set_config
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin

set_config(transform_output="pandas")

class DataFrameCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['gender_code'] = X['gender'].map({'M': 0, 'F': 1})
        X = X.drop(['gender', 'domain'], axis=1)
        X = X.drop('subject', axis=1)
        X['ecg_mean'] = X['ecg'].apply(lambda x: x.mean())
        X['ppg_mean'] = X['ppg'].apply(lambda x: x.mean())
        X['var_ecg'] = X['ecg'].apply(lambda x: x.var())
        X['var_ppg'] = X['ppg'].apply(lambda x: x.var())
        X = X.drop(['ecg', 'ppg'], axis=1)
        return X

class IgnoreDomain(RandomForestRegressor):
    def fit(self, X, y):
        X = X[y != -1]
        y = y[y != -1]
        return super().fit(X, y)

def get_estimator():
    return make_pipeline(
        DataFrameCleaner(),
        make_column_transformer(
            ("passthrough", ["age", "gender_code", "ecg_mean", "ppg_mean", "var_ecg", "var_ppg"])
        ),
        IgnoreDomain(n_estimators=15, max_depth=20, min_samples_leaf=3, random_state=42)
    )
