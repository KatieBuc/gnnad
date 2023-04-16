# -*- coding: utf-8 -*-
"""Tests gnnad for alignment with original code."""

__author__ = ["KatieBuc"]

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_random_state

from gnnad.graphanomaly import GNNAD

random_state = 245
rng = check_random_state(random_state)

# generate multivariate data
cov = [[0.5, 0.3, 0], [0.3, 1.0, 0], [0, 0, 0.8]]
mean = [1, 3, 10]
X_train = (
    pd.DataFrame(rng.multivariate_normal(mean=mean, cov=cov, size=2000))
    .ewm(span=2)
    .mean()
)
X_test = (
    pd.DataFrame(rng.multivariate_normal(mean=mean, cov=cov, size=1000))
    .ewm(span=2)
    .mean()
)

# add anomalies to the test set
X_test.iloc[342:356, :] *= 2
X_test.iloc[752:772, 0:2] *= 0.01

# anomaly labels
y_test = np.zeros(len(X_test))
y_test[342:356] = 1
y_test[752:772] = 1


def test_results():
    model = GNNAD(
        shuffle_train=False,
        topk=1,
        epoch=1,
        slide_win=2,
        dim=10,
        save_model_name="test",
        use_deterministic=True,
    )
    fitted_model = model.fit(X_train, X_test, y_test)

    precision_expected = 0.8235294117647058
    precision_actual = fitted_model.precision
    assert np.allclose(precision_actual, precision_expected, atol=0.01)

    recall_expected = 0.4117647058823529
    recall_actual = fitted_model.recall
    assert np.allclose(recall_actual, recall_expected, atol=0.01)

    f1_expected = 0.5384615384615384
    f1_actual = fitted_model.f1
    assert np.allclose(f1_actual, f1_expected, atol=0.01)

    test_avg_loss_expected = 40.152103900909424
    test_avg_loss_actual = fitted_model.test_avg_loss
    assert np.allclose(test_avg_loss_actual, test_avg_loss_expected, atol=1)


def test_weights():
    model = GNNAD(
        shuffle_train=False,
        topk=1,
        epoch=1,
        slide_win=2,
        dim=10,
        save_model_name="test",
        use_deterministic=True,
    )
    fitted_model = model.fit(X_train, X_test, y_test)
    actual_embedding_weight = fitted_model.model.embedding.weight.detach().numpy()
    expected_embedding_weight = np.array(
        [
            [
                0.01426902,
                0.04613796,
                0.07799654,
                0.12409672,
                0.01894209,
                -0.15133338,
                0.14663844,
                -0.30633965,
                -0.18445346,
                -0.08212205,
            ],
            [
                -0.15535806,
                -0.11062703,
                -0.25619,
                -0.06726643,
                0.06978236,
                -0.20897247,
                -0.01918628,
                0.22337313,
                -0.03551229,
                0.0117917,
            ],
            [
                -0.02427635,
                0.06100012,
                0.2010701,
                0.30254236,
                0.19782569,
                0.3002309,
                -0.02287012,
                -0.28707066,
                -0.15075496,
                0.21234797,
            ],
        ],
    )
    assert np.isclose(actual_embedding_weight, expected_embedding_weight).all()

    actual_bn_outlayer_in_weight = (
        fitted_model.model.bn_outlayer_in.weight.detach().numpy()
    )
    expected_bn_outlayer_in_weight = np.array(
        [
            0.9969977,
            0.99700737,
            0.9970053,
            1.0029985,
            0.99700546,
            1.0029914,
            0.99700016,
            1.0029731,
            1.0030023,
            0.99699783,
        ],
    )
    assert np.isclose(
        actual_bn_outlayer_in_weight,
        expected_bn_outlayer_in_weight,
    ).all()

    actual_bn_outlayer_in_bias = fitted_model.model.bn_outlayer_in.bias.detach().numpy()
    expected_bn_outlayer_in_bias = np.array(
        [
            -0.00300243,
            -0.00299278,
            -0.00299642,
            0.00299892,
            -0.00299407,
            0.00299447,
            -0.00299962,
            0.00300115,
            0.00300197,
            -0.00299924,
        ]
    )
    assert np.isclose(
        actual_bn_outlayer_in_bias,
        expected_bn_outlayer_in_bias,
    ).all()


def _test_fitted_model():
    """This model is tested against _[1], with the above generated code.
    To run the _original_ code, this data needs to be saved as csv's locally.

    X_test['attack'] = y_test
    X_train.to_csv('data/unit_test/train.csv')
    X_test.to_csv('data/unit_test/test.csv')

    Then, in the terminal run:
    python main.py -dataset unit_test -device cpu -topk 2 -epoch 10

    References
    ----------
    _[1] https://github.com/d-ailin/GDN
    """

    model = GNNAD(shuffle_train=False, topk=2, epoch=10, save_model_name="test")
    fitted_model = model.fit(X_train, X_test, y_test)

    f1_actual = fitted_model.f1
    precision_actual = fitted_model.precision
    recall_actual = fitted_model.recall

    f1_expected = 0.5384615384615384
    precision_expected = 0.8235294117647058
    recall_expected = 0.4117647058823529

    assert np.allclose(f1_actual, f1_expected)
    assert np.allclose(precision_actual, precision_expected)
    assert np.allclose(recall_actual, recall_expected)


def _test_fitted_model_val():
    """Test model detection for threshold as maximum A(t) from validation data.

    This model is tested against _[1], with the above generated code.
    To run the _original_ code, this data needs to be saved as csv's locally.

    X_test['attack'] = y_test
    X_train.to_csv('data/unit_test/train.csv')
    X_test.to_csv('data/unit_test/test.csv')

    Then, in the terminal run:
    python main.py -dataset unit_test -device cpu -topk 2 -epoch 10 -report val

    References
    ----------
    _[1] https://github.com/d-ailin/GDN
    """

    model = GNNAD(
        shuffle_train=False,
        topk=2,
        epoch=10,
        threshold_type="max_validation",
        save_model_name="test",
    )
    fitted_model = model.fit(X_train, X_test, y_test)

    f1_actual = fitted_model.f1
    precision_actual = fitted_model.precision
    recall_actual = fitted_model.recall

    f1_expected = 0.23728813559322035
    precision_expected = 0.16666666666666666
    recall_expected = 0.4117647058823529

    assert np.allclose(f1_actual, f1_expected)
    assert np.allclose(precision_actual, precision_expected)
    assert np.allclose(recall_actual, recall_expected)
