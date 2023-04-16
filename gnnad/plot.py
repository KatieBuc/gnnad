# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def plot_test_anomalies(
    X_test,
    ANOMS,
    fig_cols=1,
    color="r",
    s=10,
    alpha=0.4,
    fontsize="large",
    loc="left",
    figsize=(20, 20),
):
    n_nodes = X_test.shape[1]
    fig_rows = (
        n_nodes // fig_cols if n_nodes % fig_cols == 0 else n_nodes // fig_cols + 1
    )
    fig, axs = plt.subplots(fig_rows, fig_cols, figsize=figsize)

    for i, ax in enumerate(axs.flat):
        if i >= n_nodes:
            break

        node_num = X_test.columns[i]
        ax.plot(X_test[node_num], alpha=0.5)

        for anom_type in ANOMS.keys():
            if node_num in ANOMS[anom_type].keys():
                ax.scatter(
                    ANOMS[anom_type][node_num],
                    X_test[node_num][ANOMS[anom_type][node_num]].values,
                    color=color,
                    s=s,
                    alpha=alpha,
                )
        ax.set_title(node_num, fontsize=fontsize, loc=loc)

    fig.tight_layout()
    plt.subplots_adjust(right=0.9)
    plt.show()


def plot_predictions(
    fitted_model,
    X_test,
    ANOMS,
    preds=None,
    figsize=(7, 10),
    fig_cols=1,
):
    fig_rows = (
        fitted_model.n_nodes // fig_cols
        if fitted_model.n_nodes % fig_cols == 0
        else fitted_model.n_nodes // fig_cols + 1
    )
    fig_rows += 1

    fig, axs = plt.subplots(fig_rows, fig_cols, figsize=figsize)
    plt_idx = X_test.index[fitted_model.slide_win :]

    for i, ax in enumerate(axs.flat):
        if i >= fitted_model.n_nodes:
            break
        test_predict, test_ground = fitted_model.test_result[:2, :, i]
        if i == 0:
            ax.plot(plt_idx, test_predict.T, c="slateblue", label="pred level")
            ax.plot(plt_idx, test_ground.T, c="mediumaquamarine", label="true level")
        else:
            ax.plot(plt_idx, test_predict.T, c="slateblue")
            ax.plot(plt_idx, test_ground.T, c="mediumaquamarine")

        node_num = X_test.columns[i]

        for anom_type in ANOMS.keys():
            if node_num in ANOMS.keys():
                _idx = [
                    i
                    for i in ANOMS[anom_type][node_num]
                    if i in X_test.index[fitted_model.slide_win :]
                ]
                ax.scatter(
                    _idx, X_test=[node_num][_idx].values, color="r", s=10, alpha=0.4
                )

        ax.set_title(X_test.columns[i], fontsize="large", loc="center")

    p = pd.concat(
        [
            pd.DataFrame(fitted_model.test_labels),
            pd.DataFrame(fitted_model.pred_labels if preds is None else preds),
        ],
        axis=1,
    )
    p.index = X_test[fitted_model.slide_win :].index
    p.columns = ["true", "predicted"]

    p.index = pd.to_datetime(p.index)
    axs[-1].plot(p.predicted, c="darkorange", label="pred anom")
    axs[-1].scatter(
        p.true[p.true == 1].index,
        p.true[p.true == 1].values,
        color="r",
        s=10,
        alpha=0.4,
        label="true anom",
    )
    axs[-1].set_title("anomalies", fontsize="large", loc="center")
    axs[-1].set_yticks([0, 1])
    if isinstance(X_test.index, pd.DatetimeIndex):
        axs[-1].xaxis.set_major_locator(mdates.DayLocator(interval=7))

    for ax in axs:
        if ax != axs[-1]:
            ax.set_xticks([])

    fig.legend(bbox_to_anchor=(0.92, 1.03))
    fig.tight_layout()
    plt.subplots_adjust(right=0.9)
    plt.show()


def plot_sensor_error_scores(
    fitted_model, X_test, fig_cols=1, color="r", s=10, alpha=0.4, figsize=(20, 20)
):
    n_nodes = fitted_model.n_nodes
    fig_rows = (
        n_nodes // fig_cols if n_nodes % fig_cols == 0 else n_nodes // fig_cols + 1
    )

    fig, axs = plt.subplots(fig_rows, fig_cols, figsize=figsize)

    sensor_preds = fitted_model.get_sensor_preds()
    sensor_preds = pd.DataFrame(
        sensor_preds.T,
        index=X_test.index[fitted_model.slide_win :],
        columns=X_test.columns,
    )

    for i, ax in enumerate(axs.flat):
        if i >= n_nodes:
            break

        node_num = X_test.columns[i]
        err_score = pd.DataFrame(
            fitted_model.test_err_scores[i, :],
            index=X_test.index[fitted_model.slide_win :],
            columns=[node_num],
        )
        ax.plot(err_score, alpha=0.5)

        ax.scatter(
            sensor_preds[node_num][sensor_preds[node_num].astype(bool)].index,
            err_score[sensor_preds[node_num].astype(bool)].values,
            color=color,
            s=s,
            alpha=alpha,
        )
        ax.set_title(node_num, fontsize="small", loc="left")

    fig.tight_layout()
    plt.subplots_adjust(right=0.9)
    plt.show()
