import sklearn.metrics as metrics
import numpy as np
import collections
import json
import os
import pandas as pd

"""
collections of sklear metrics to evaluate
the predicted results of the model.
"""


def metrics_sklearn(y_true, y_pred, threshold=0.5):
    """
    fun runs several metrics from sklearn
    and returns a dictionary of values.

    paramenters:
    y_true=real values
    y_pred=predected values
    threshold=threshold score for positive class
    """
    metrics_df = collections.defaultdict()

    y_pred_class = np.where(y_pred >= threshold, 1, 0)

    metrics_df["y_true"] = y_true.tolist()
    metrics_df["y_pred"] = y_pred.tolist()
    metrics_df["y_pred_class"] = y_pred_class.tolist()
    metrics_df["threshold"] = threshold

    (
        metrics_df["precision"],
        metrics_df["recall"],
        metrics_df["fbeta_score"],
        _,
    ) = metrics.precision_recall_fscore_support(
        y_true, y_pred_class, beta=1.0, average="samples"
    )

    metrics_df["accuracy"] = metrics.accuracy_score(y_true, y_pred_class)

    # metrics_df["auc_score"] = metrics.roc_auc_score(y_true, y_pred)

    # # metrics_df["average_prec_score"] = metrics.average_precision_score(y_true, y_pred)

    # metrics_df["f_beta_score"] = metrics.fbeta_score(
    #     y_true, y_pred_class, 1.0, average=None
    # ).tolist()

    # metrics_df["precision_score"] = metrics.precision_score(
    #     y_true, y_pred_class, average=None
    # ).tolist()

    # metrics_df["recall_score"] = metrics.recall_score(
    #     y_true, y_pred_class, average=None
    # ).tolist()

    # metrics_df["r2_score"] = metrics.r2_score(y_true, y_pred).tolist()
    return metrics_df


def dump_metrics(metrics_dict, path, file_name):
    """
    fun dumps the metrics to a json file for
    future usage.

    paramenters:
    metrics_dict=metrics
    path=output directory path
    file_name=output file name
    """
    file_path = os.path.join(path, f"{file_name}")
    with open(file_path, "w") as outfile:
        json.dump(metrics_dict, outfile)
    return file_path


def evaluate_model(
    y_true, y_pred, threshold, output_dir=None, json_name="metrics.json"
):
    """
    fun runs metrics evaluations
    and dump json file.

    parameters:
    y_true=real values
    y_pred=predected values
    threshold=threshold score for positive class
    output_dir=json output dir
    json_name=json file name
    """
    if not output_dir:
        output_dir = os.getcwd()
    metrics = metrics_sklearn(y_true, y_pred)
    dump_metrics(metrics, output_dir, json_name)
    return json_name


if __name__ == "__main__":
    np.random.seed(1989)
    y_true = np.random.randint(low=0, high=2, size=100)
    y_pred = np.random.uniform(low=0, high=1, size=100)
    scores = evaluate_model(
        y_true, y_pred, 0.5, output_dir=None, json_name="metrics.json"
    )
