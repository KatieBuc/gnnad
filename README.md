# Graph Neural Network-Based Anomaly Detection (GNNAD)

`gnnad` is a package for anomaly detection on multivariate time series data.

This model builds on the recently-proposed Graph Deviation Network (GDN)[^2], a graph neural network model that uses embeddings to capture inter-sensor relationships as a learned graph, and employs graph attention-based forecasting to predict future sensor behaviour. Anomalies are flagged when the error scores are above a calculated threshold value. By learning the interdependencies among variables and predicting based on the typical patterns of the system, this approach is able to detect deviations when the expected spatial dependencies are disrupted. As such, GDN offers the ability to detect even the small-deviation anomalies generally overlooked by other distance based and density based anomaly detection methods for time series.

This package accompanies work that further develops this model[^1], and introduces an alternate anomaly threshold criteria based on the learned graph, with the improved ability to detect anomalies in multivariate timeseries data.

## Quick start

As an example we'll explore data collected within river network system. In particular data collected on the Herbert river network, at these sensor locations:

<p align="center">
<img src="https://user-images.githubusercontent.com/34525024/232662278-bc6973ae-6ccf-443d-99d4-204eada127d6.JPG" width="35%" height="35%" alt="Herbert river sensor locations" title="">
 </p>

The sensors measure water level, from within the river. Assuming we have pre-processed the data, we instantiate and fit the model:

```python
from gnnad.graphanomaly import GNNAD

# run model
model = GNNAD(threshold_type="max_validation", topk=6, slide_win=200)
fitted_model = model.fit(X_train, X_test, y_test)

# the predicted values can be accessed here
test_predict = fitted_model.test_result[0, :, i]
```

We can visualise the predicted values vs. actual values, with helper functions in the plot module.
<p align="center">
<img src="https://user-images.githubusercontent.com/34525024/232661014-99ebb7c0-7e4a-4f54-b09a-fedb5c5bbaf1.jpg" width="40%" height="40%" alt="Herbert river sensor locations" title="">
</p>
  
Note that some locations closer to the outlet are influenced by tidal patterns. The error scores that are obtained from this forecasting model are then transformed and, if they exceed the calculated threshold, flagged as an anomaly. The bottom indicates if any sensor flagged an anomaly, and compares this to the ground truth labels, for the test data.

The performance of the anomaly detection classification model can be analysed by:

```python
fitted_model.print_eval_metrics()
>>> recall: 30.4
>>> precision: 59.3
>>> accuracy: 49.7
>>> specificity: 73.9
>>> f1: 40.2
```

Check out full details in the [example notebook](example_herbert.ipynb)

## Installation

`gnnad` is compatible with python versions 3.8, 3.9, 3.10 and 3.11. You can install the latest release with pip: 

```bash
pip install gnnad
```

If you would like the latest development version you can install directly from github:

```bash
pip install https://github.com/KatieBuc/gnnad.git
```

## Developer installation

You'll need [poetry](https://python-poetry.org/docs/#installation). Once you have it installed and
cloned the repo you can install with (from the repo directory):

```bash
poetry install
```

[^1]: Buchhorn, Katie, et al. _"Graph Neural Network-Based Anomaly Detection for River Network Systems"_ Under review.
[^2]: Deng, Ailin, and Bryan Hooi. _"Graph neural network-based anomaly detection in multivariate time series."_ Proceedings of the AAAI conference on artificial intelligence. Vol. 35. No. 5. 2021.


## Citation
MLA:
```
Buchhorn, Katie, et al. "Graph Neural Network-Based Anomaly Detection for River Network Systems"
arXiv preprint arXiv:2304.09367 (2023).
```
BibTeX:
```
@article{buchhorn2023graph,
  title={Graph Neural Network-Based Anomaly Detection for River Network Systems},
  author={Buchhorn, Katie and Mengersen, Kerrie and Santos-Fernandez, Edgar and Salomone, Robert},
  journal={arXiv preprint arXiv:2304.09367},
  year={2023}
}
```
