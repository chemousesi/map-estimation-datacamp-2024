from sklearn import set_config
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

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

        def dominant_frequency(signal):
            fft_vals = np.fft.rfft(signal)
            freqs = np.fft.rfftfreq(len(signal), d=1/100)
            dominant_freq = freqs[np.argmax(np.abs(fft_vals))]
            return dominant_freq

        X['ecg_dominant_freq'] = X['ecg'].apply(dominant_frequency)
        X['ppg_dominant_freq'] = X['ppg'].apply(dominant_frequency)

        # def wavelet_energy(signal, wavelet='db4', level=3):
        #     coeffs = pywt.wavedec(signal, wavelet, level=level)
        #     energy = sum(np.linalg.norm(c)**2 for c in coeffs[1:])
        #     return energy

        # X['ecg_wavelet_energy'] = X['ecg'].apply(wavelet_energy)
        # X['ppg_wavelet_energy'] = X['ppg'].apply(wavelet_energy)

        def compute_auc(signal):
            return np.trapz(signal, dx=1/100)

        X['ecg_auc'] = X['ecg'].apply(compute_auc)
        X['ppg_auc'] = X['ppg'].apply(compute_auc)

        def compute_derivatives(signal):
            first_derivative = np.gradient(signal, 1/100)
            second_derivative = np.gradient(first_derivative, 1/100)
            return first_derivative.mean(), second_derivative.mean()

        X[['ppg_slope', 'ppg_acceleration']] = X['ppg'].apply(
            lambda x: compute_derivatives(x)
        ).apply(pd.Series)

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
            ("passthrough", [
                "age", "gender_code", "ecg_mean", "ppg_mean",
                "var_ecg", "var_ppg",
                "ecg_dominant_freq", "ppg_dominant_freq",
                "ecg_auc", "ppg_auc",
                "ppg_slope", "ppg_acceleration"
            ])
        ),
        IgnoreDomain(n_estimators=15, max_depth=20,
                     min_samples_leaf=3, random_state=42)
    )
