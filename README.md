# Graph Neural Network-Based Anomaly Detection (GNNAD)

gnnad is a package for anomaly detection on multivariate time series data. This model builds on the recently-proposed Graph Deviation Network (GDN) [2], a  graph neural network model that uses embeddings to capture inter-sensor relationships as a learned graph, and employs graph attention-based forecasting to predict future sensor behaviour. Anomalies are flagged when the error scores are above a calculated threshold value. By learning the interdependencies among variables and predicting based on the typical patterns of the system, this approach is able to detect deviations when the expected spatial dependencies are disrupted. As such, GDN offers the ability to detect even the small-deviation anomalies generally overlooked by other distance based and density based anomaly detection methods for time series.

This package accompanies work that further develops this model [1], and introduces an alternate anomaly threshold criteria based on the learned graph, with the improved ability to detect anomalies in multivariate timeseries data. The example we explore is of data collected within river network systems.

## Quick start

Consider data collected on the Herbert river network, at these locations. We see that some locations closer to the outlet are influenced by tidal patterns.

<img src="https://github.com/KatieBuc/gnnad/files/11257391/herbert_ssn.pdf " alt="Herbert river sensor locations" title="">

Assuming we have pre-processed the data, we instantiate and fit the model:
```
# run model
model = GNNAD(threshold_type="max_validation", topk=6, slide_win=200)
fitted_model = model.fit(X_train, X_test, y_test)

# the predicted values can be accessed here
test_predict = fitted_model.test_result[0, :, i]
```

We can visualise the predicted values vs. actual values, with helper functions in the plot module.

![Herbert](https://user-images.githubusercontent.com/34525024/232661014-99ebb7c0-7e4a-4f54-b09a-fedb5c5bbaf1.jpg)

The error scores that are obtained from this forecasting model are then transformed, and if they exceed a calculated threshold, then flagged as an anomaly.

The performance of the anmaly detection classification model can be called:
```
fitted_model.print_eval_metrics()
>>> recall: 30.4
>>> precision: 59.3
>>> accuracy: 49.7
>>> specificity: 73.9
>>> f1: 40.2
```

Check out full details in the [example notebook](example_herbert.ipynb)

## Installation

```bash
pip install https://github.com/KatieBuc/gnnad.git
```

Once the project is release on PyPI it should be as simple as:

```bash
pip install gnnad
```

## Developer installation

You'll need [poetry](https://python-poetry.org/docs/#installation). Once you have it installed and
cloned the repo you can install with (from the repo directory):

```bash
poetry install
```

## References

[1] Buchhorn, Katie, et al. "Graph Neural Network-Based Anomaly Detection for River Network Systems" Under review.

[2] Deng, Ailin, and Bryan Hooi. "Graph neural network-based anomaly detection in multivariate time series." Proceedings of the AAAI conference on artificial intelligence. Vol. 35. No. 5. 2021.


## Citation

```
Buchhorn, Katie, et al. "Graph Neural Network-Based Anomaly Detection for River Network Systems" Under review.
```
