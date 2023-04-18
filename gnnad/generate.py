# -*- coding: utf-8 -*-
"""Data generation and anomaly generation."""

import collections
import itertools
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gstools import SRF, Gaussian

__author__ = ["KatieBuc"]


class GenerateGaussian:
    """
    Generate smoothened random Gaussian field and sample locations.
    """

    def __init__(
        self,
        seed=1,
        n_obs=20,
        x_lim=(0, 10),
        y_lim=(0, 10),
        T=400,
        n_lags=3,
        weight=[0.5, 0.25, 0.15, 0.1],
        dim=2,
        var=1,
        len_scale=2,
    ):
        """
        -----------
        seed: int
            Seed for random number generator.
        n_obs: int
            Number of observation points to be sampled.
        x_lim: tuple
            Tuple of x-axis lower and upper limits for field generation.
        y_lim: tuple
            Tuple of y-axis lower and upper limits for field generation.
        T: int
            Number of time points to generate for each observation point.
        n_lags: int
            Number of lags for the field smoothing.
        weight: list
            Weights for the smoothing function.
        dim: int
            Number of dimensions of the field.
        var: float
            Variance of the Gaussian distribution.
        len_scale: float
            Length scale of the Gaussian distribution.
        """
        self.seed = seed
        self.n_obs = n_obs
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.T = T
        self.n_lags = n_lags
        self.weight = weight
        self.dim = dim
        self.var = var
        self.len_scale = len_scale

    def weighted_average(self, field, weights, start, end):
        return [
            sum(weights[j] * field[i - j] for j in range(len(weights)))
            for i in range(start, end)
        ]

    def generate_field(self):
        model = Gaussian(dim=self.dim, var=self.var, len_scale=self.len_scale)
        srf = SRF(model)

        self.x_field = np.arange(self.x_lim[0], self.x_lim[1], 0.1)
        self.y_field = np.arange(self.y_lim[0], self.y_lim[1], 0.1)

        start_idx, end_idx = self.n_lags, self.T + self.n_lags
        self.field = np.array(
            [
                srf.structured([self.x_field, self.y_field], seed=i)
                for i in range(end_idx)
            ]
        )
        self.smooth_field = np.array(
            self.weighted_average(self.field, self.weight, start_idx, end_idx)
        )

    def sample_locations(self):
        random.seed(self.seed)
        self.x_idxs = random.choices(
            range(len(self.x_field)), k=self.n_obs
        )  # with replacement
        self.y_idxs = random.sample(
            range(len(self.y_field)), k=self.n_obs
        )  # without replacement

    def get_dataframe(self):
        return pd.DataFrame(
            [
                [t, i, self.smooth_field[t][x_idx][y_idx]]
                for i, (x_idx, y_idx) in enumerate(zip(self.x_idxs, self.y_idxs))
                for t in range(self.T)
            ],
            columns=["t", "id", "X"],
        )

    def pivot_dataframe(self, df):
        df = df.pivot(index="t", columns="id")
        df.columns = df.columns.get_level_values(1)
        df.index.name = df.columns.name = ""
        return df

    def generate(self):
        self.generate_field()
        self.sample_locations()
        df = self.get_dataframe()
        df = self.pivot_dataframe(df)
        return df

    def field_plot(self, n_times=3, figsize=(10, 7)):
        fig, ax = plt.subplots(1, n_times, sharex=True, sharey=True, figsize=figsize)
        ax = ax.flatten()
        for i in range(n_times):
            ax[i].imshow(self.smooth_field[i].T, origin="lower")
            ax[i].scatter(self.x_idxs, self.y_idxs, c="grey")
            ax[i].text(3, 2, f"t={i + 1}", fontsize=18, c="white")

        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        fig.tight_layout()


class GenerateAnomaly:
    """
    Generate anomalies for timeseries
    """

    def __init__(self, X, var_scale=10, drift_delta=0.85):
        """
        Parameters:
        ----------
        X : pandas.DataFrame
            A pandas DataFrame containing the time series data.
        var_scale : float, optional
            The variance scale used for generating anomalies.
        drift_delta : float, optional
            The drift delta used for generating anomalies.
        """
        self.X = X
        self.var_scale = var_scale
        self.drift_delta = drift_delta

        # to be updated
        self._X = X
        self.ANOMS = collections.defaultdict(dict)

    def variability(self, size):
        return np.random.normal(0, self.var_scale, size=size)

    def drift(self, size):
        return np.arange(
            self.drift_delta, (size + 1) * self.drift_delta, self.drift_delta
        )

    def generate(self, anomaly_func, lam=3, prop_anom=0.07, seed=45):
        num_anom = num_anom_start_idx(self.X, prop_anom, lam)
        anom_cols, anom_idxs, anom_lens = sample_anom_len(self.X, seed, num_anom, lam)

        for i in range(num_anom):
            col = anom_cols[i]
            idx_start = anom_idxs[i]
            idx_end = min(anom_idxs[i] + anom_lens[i], self.X.index[-1])

            # update dataframe
            self._X.loc[idx_start:idx_end, col] += anomaly_func(
                size=idx_end - idx_start + 1
            )

            # keep record of anomaly type, and indices
            if col in self.ANOMS[anomaly_func.__name__].keys():
                self.ANOMS[anomaly_func.__name__][col] = np.concatenate(
                    (
                        self.ANOMS[anomaly_func.__name__][col],
                        np.arange(idx_start, idx_end + 1),
                    )
                )
            else:
                self.ANOMS[anomaly_func.__name__][col] = np.arange(
                    idx_start, idx_end + 1
                )

        return self._X

    def get_labels(self):
        list2d = [list(i[1].values()) for i in self.ANOMS.items()]
        anom_idxs = np.unique(np.concatenate(list(itertools.chain(*list2d)), axis=0))
        y_test = pd.DataFrame(np.zeros(len(self.X), dtype=int), index=self.X.index)
        y_test.loc[anom_idxs] = 1

        return y_test.squeeze().values


def num_anom_start_idx(X, prop_anom, lam):
    return int(np.ceil(prop_anom * len(X) / lam))


def sample_anom_len(X, seed, num_anom, lam):
    random.seed(seed)
    np.random.seed(seed)
    anom_cols = random.choices(list(X.columns), k=num_anom)  # with replacement
    anom_idxs = random.choices(list(X.index), k=num_anom)  # with replacement
    anom_lens = np.random.poisson(lam=lam, size=num_anom)
    return anom_cols, anom_idxs, anom_lens
