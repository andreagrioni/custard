import sklearn.metrics as metrics
import numpy as np
import collections
import json
import pandas as pd

'''
collections of sklear metrics to evaluate
the predicted results of the model.
'''

def metrics_sklearn(
    y_true, y_pred, threshold = 0.5):
    '''
    fun runs several metrics from sklearn
    and returns a dictionary of values.

    paramenters:
    y_true=real values
    y_pred=predected values
    threshold=threshold score for positive class
    '''
    metrics_df = collections.defaultdict()

    y_pred_class = np.where(
        y_pred > threshold, 1, 0
        )

    metrics_df['y_true'] = y_true.tolist()
    metrics_df['y_pred'] = y_pred.tolist()
    metrics_df['threshold'] = threshold

    metrics_df['accuracy'] = metrics.accuracy_score(
        y_true, y_pred_class
    )
    
    metrics_df['auc_score'] = metrics.roc_auc_score(
        y_true, y_pred
    )

    metrics_df['average_prec_score'] = metrics.average_precision_score(
        y_true, y_pred
        )
    
    metrics_df['balanced_accuracy_score'] = metrics.balanced_accuracy_score(
        y_true, y_pred_class
    )

    metrics_df['f_beta_score'] = metrics.fbeta_score(
        y_true, y_pred_class, 1.0
    ).tolist()

    metrics_df['precision_score'] = metrics.precision_score(
        y_true, y_pred_class
    ).tolist()

    metrics_df['recall_score'] = metrics.recall_score(
        y_true, y_pred_class
    ).tolist()

    metrics_df['r2_score'] = metrics.r2_score(
        y_true, y_pred
    ).tolist()
    return metrics_df


def dump_metrics(metrics_dict, path, file_name):
    '''
    fun dumps the metrics to a json file for
    future usage.

    paramenters:
    metrics_dict=metrics
    path=output directory path
    file_name=output file name
    '''
    file_path = os.path.join(
        path, f'{file_name}.json'
        )
    with open(file_path, 'w') as outfile:
        json.dump(metrics_dict, outfile)
    return file_path


if __name__ == "__main__":
    np.random.seed(1989)
    y_true = np.random.randint(
        low=0, high=2, size=100
    )
    y_pred = np.random.uniform(
        low=0, high=1, size=100
    )

    metrics = metrics_sklearn(
            y_true, y_pred
            )
    dump_metrics(
        metrics, os.getcwd(), 'metrics.json'
        )