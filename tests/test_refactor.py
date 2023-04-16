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


def test_weights():
    model = GNNAD(
        shuffle_train=False,
        topk=1,
        epoch=1,
        slide_win=2,
        dim=10,
        save_model_name="test",
    )
    fitted_model = model.fit(X_train, X_test, y_test)
    actual_embedding_weight = fitted_model.model.embedding.weight.detach().numpy()
    expected_embedding_weight = np.array(
        [
            [
                0.01426902,
                0.04613796,
                0.07799935,
                0.12409672,
                0.01894209,
                -0.15130585,
                0.14663096,
                -0.30634218,
                -0.18444178,
                -0.08214602,
            ],
            [
                -0.15535805,
                -0.11062703,
                -0.2561871,
                -0.06726643,
                0.06978149,
                -0.2090014,
                -0.01923222,
                0.22337098,
                -0.0355087,
                0.01179178,
            ],
            [
                -0.02427622,
                0.06099382,
                0.2010701,
                0.3025389,
                0.1978267,
                0.3002309,
                -0.02287012,
                -0.28707218,
                -0.15046325,
                0.2123317,
            ],
        ]
    )
    assert np.isclose(actual_embedding_weight, expected_embedding_weight).all()

    actual_bn_outlayer_in_weight = (
        fitted_model.model.bn_outlayer_in.weight.detach().numpy()
    )
    expected_bn_outlayer_in_weight = np.array(
        [
            0.99700457,
            0.9969961,
            0.9970133,
            1.0029953,
            0.99700665,
            1.0030003,
            0.996998,
            1.0029932,
            1.0030003,
            0.9969989,
        ]
    )
    assert np.isclose(
        actual_bn_outlayer_in_weight,
        expected_bn_outlayer_in_weight,
    ).all()
    actual_bn_outlayer_in_bias = fitted_model.model.bn_outlayer_in.bias.detach().numpy()
    expected_bn_outlayer_in_bias = np.array(
        [
            -0.00299567,
            -0.00300373,
            -0.00299501,
            0.0029959,
            -0.00299488,
            0.00300092,
            -0.00300224,
            0.00300236,
            0.00299983,
            -0.00299955,
        ]
    )
    assert np.isclose(
        actual_bn_outlayer_in_bias,
        expected_bn_outlayer_in_bias,
    ).all()

    actual_gnn_att_i = fitted_model.model.gnn_layers[0].gnn.att_i.detach().numpy()
    expected_gnn_att_i = np.array(
        [
            [
                [
                    -0.12153649,
                    -0.33807585,
                    0.28401357,
                    -0.43744513,
                    0.2707457,
                    0.37349015,
                    0.52870613,
                    0.27615172,
                    -0.73096794,
                    -0.47909442,
                ]
            ]
        ]
    )
    assert np.isclose(actual_gnn_att_i, expected_gnn_att_i).all()

    actual_gnn_att_j = fitted_model.model.gnn_layers[0].gnn.att_j.detach().numpy()
    expected_gnn_att_j = np.array(
        [
            [
                [
                    0.36876863,
                    0.15457928,
                    -0.5761302,
                    -0.4252709,
                    0.69478935,
                    0.49764746,
                    -0.3220259,
                    -0.18588156,
                    -0.7035403,
                    -0.01327479,
                ]
            ]
        ]
    )

    assert np.isclose(actual_gnn_att_j, expected_gnn_att_j).all()

    actual_gnn_att_em_i = fitted_model.model.gnn_layers[0].gnn.att_em_i.detach().numpy()
    expected_gnn_att_em_i = np.array(
        [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
    )
    assert np.isclose(
        actual_gnn_att_em_i,
        expected_gnn_att_em_i,
    ).all()

    actual_gnn_att_em_j = fitted_model.model.gnn_layers[0].gnn.att_em_j.detach().numpy()
    expected_gnn_att_em_j = np.array(
        [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
    )
    assert np.isclose(
        actual_gnn_att_em_j,
        expected_gnn_att_em_j,
    ).all()


def _test_weights():
    actual_gnn_bias = fitted_model.model.gnn_layers[0].gnn.bias.detach().numpy()
    expected_gnn_bias = np.array(
        [
            2.7264672e-05,
            2.1879865e-04,
            3.1615925e-04,
            -1.8837597e-04,
            1.5300469e-04,
            9.6737960e-04,
            -4.7895181e-04,
            -1.5088532e-03,
            1.5319383e-03,
            -2.8946530e-03,
        ]
    )
    assert np.isclose(actual_gnn_bias, expected_gnn_bias).all()

    actual_gnn_lin_weight = (
        fitted_model.model.gnn_layers[0].gnn.lin.weight.detach().numpy()
    )
    expected_gnn_lin_weight = np.array(
        [
            [0.02404292, 0.16679624],
            [0.43820742, 0.67942816],
            [-0.54253733, -0.26150995],
            [0.28046, 0.5833108],
            [0.61481285, 0.6244394],
            [0.13774624, -0.61786443],
            [0.067935, -0.439477],
            [-0.6619868, 0.6252629],
            [0.53467536, -0.70833904],
            [0.12937956, -0.12208727],
        ]
    )
    assert np.isclose(
        actual_gnn_lin_weight,
        expected_gnn_lin_weight,
    ).all()

    actual_bn_weight = fitted_model.model.gnn_layers[0].bn.weight.detach().numpy()
    expected_bn_weight = np.array(
        [
            0.99699503,
            0.9970028,
            0.99705946,
            1.0029914,
            0.99700224,
            1.0029453,
            0.9969974,
            1.0003419,
            1.0029472,
            0.9970299,
        ]
    )
    assert np.isclose(actual_bn_weight, expected_bn_weight).all()

    actual_bn_bias = fitted_model.model.gnn_layers[0].bn.bias.detach().numpy()
    expected_bn_bias = np.array(
        [
            -0.00289865,
            -0.00299934,
            -0.00299644,
            0.0029283,
            -0.00281849,
            0.00295435,
            -0.00294909,
            0.00300207,
            0.00298393,
            -0.00292938,
        ]
    )
    assert np.isclose(actual_bn_bias, expected_bn_bias).all()

    actual_mlp_weight = fitted_model.model.out_layer.mlp[0].weight.detach().numpy()
    expected_mlp_weight = np.array(
        [
            [
                -0.2351357,
                -0.24093145,
                -0.01442307,
                0.05047624,
                -0.12650947,
                0.19063334,
                -0.18943678,
                0.28990906,
                0.21971291,
                -0.2636672,
            ]
        ]
    )
    assert np.isclose(actual_mlp_weight, expected_mlp_weight).all()

    actual_mlp_bias = fitted_model.model.out_layer.mlp[0].bias.detach().numpy()
    expected_mlp_bias = np.array([-0.07570441])
    assert np.isclose(actual_mlp_bias, expected_mlp_bias).all()


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
