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
                0.01179179,
            ],
            [
                -0.02427622,
                0.06099382,
                0.2010701,
                0.302539,
                0.19782667,
                0.3002309,
                -0.02287012,
                -0.28707218,
                -0.15046325,
                0.2123317,
            ],
        ]
    )

    assert np.isclose(
        fitted_model.model.embedding.weight.detach().numpy(), expected_embedding_weight
    ).all()

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
        fitted_model.model.bn_outlayer_in.weight.detach().numpy(),
        expected_bn_outlayer_in_weight,
    ).all()

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
        fitted_model.model.bn_outlayer_in.bias.detach().numpy(),
        expected_bn_outlayer_in_bias,
    ).all()

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

    assert np.isclose(
        fitted_model.model.gnn_layers[0].gnn.att_i.detach().numpy(), expected_gnn_att_i
    ).all()

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

    assert np.isclose(
        fitted_model.model.gnn_layers[0].gnn.att_j.detach().numpy(), expected_gnn_att_j
    ).all()

    expected_gnn_att_em_i = np.array(
        [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
    )

    assert np.isclose(
        fitted_model.model.gnn_layers[0].gnn.att_em_i.detach().numpy(),
        expected_gnn_att_em_i,
    ).all()

    expected_gnn_att_em_j = np.array(
        [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
    )

    assert np.isclose(
        fitted_model.model.gnn_layers[0].gnn.att_em_j.detach().numpy(),
        expected_gnn_att_em_j,
    ).all()

    expected_gnn_bias = np.array(
        [
            1.0915086e-04,
            -8.6545211e-04,
            -2.0682339e-05,
            -5.1743365e-05,
            3.6255823e-04,
            -7.9241034e-04,
            -1.2985674e-04,
            1.1404270e-03,
            1.5715663e-03,
            -1.1722990e-03,
        ]
    )

    assert np.isclose(
        fitted_model.model.gnn_layers[0].gnn.bias.detach().numpy(), expected_gnn_bias
    ).all()

    expected_gnn_lin_weight = np.array(
        [
            [0.02403709, 0.16680013],
            [0.44081104, 0.67682326],
            [-0.54256153, -0.2614859],
            [0.28046358, 0.583309],
            [0.6126126, 0.62663656],
            [0.13772677, -0.61788493],
            [0.06804692, -0.43936962],
            [-0.66194206, 0.62530875],
            [0.5346754, -0.70833844],
            [0.12936948, -0.12209835],
        ]
    )

    assert np.isclose(
        fitted_model.model.gnn_layers[0].gnn.lin.weight.detach().numpy(),
        expected_gnn_lin_weight,
    ).all()

    expected_bn_weight = np.array(
        [
            0.9969971,
            0.9969953,
            0.99704933,
            1.002982,
            0.99700177,
            1.0029645,
            0.9969969,
            1.0003407,
            1.0029299,
            0.9970252,
        ]
    )

    assert np.isclose(
        fitted_model.model.gnn_layers[0].bn.weight.detach().numpy(), expected_bn_weight
    ).all()

    expected_bn_bias = np.array(
        [
            -0.00292829,
            -0.0029352,
            -0.00299965,
            0.00298119,
            -0.00281256,
            0.00300103,
            -0.00300342,
            0.0030018,
            0.00299224,
            -0.0029569,
        ]
    )

    assert np.isclose(
        fitted_model.model.gnn_layers[0].bn.bias.detach().numpy(), expected_bn_bias
    ).all()

    expected_mlp_weight = np.array(
        [
            [
                -0.23514244,
                -0.24092038,
                -0.01442866,
                0.05047251,
                -0.12651062,
                0.19064255,
                -0.18943474,
                0.28992897,
                0.21971077,
                -0.26366824,
            ]
        ]
    )

    assert np.isclose(
        fitted_model.model.out_layer.mlp[0].weight.detach().numpy(), expected_mlp_weight
    ).all()

    expected_mlp_bias = np.array([-0.07570465])

    assert np.isclose(
        fitted_model.model.out_layer.mlp[0].bias.detach().numpy(), expected_mlp_bias
    ).all()


def test_fitted_model():
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


def test_fitted_model_val():
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
