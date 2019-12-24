from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import misc
import logging

"""
this module creates input dataset
for custard
"""


def watson_crick(x_nt, y_nt, alphabet=None):
    """
    fun assigns 1 if input string
    is in alphabet, otherwise
    it returns 0.
    parameters:
    x_nt = nucleotide on x axis
    y_nt = nucleotide on y axis
    alphabet = dict of nt_pair:score
    """
    if not alphabet:
        alphabet = {"AT": 1, "TA": 1, "GC": 1, "CG": 1}
    pair = x_nt + y_nt
    return alphabet.get(pair, 0)


def one_hot_encoding(df, shape):
    """
    fun transform input database to
    one hot encoding array.

    paramenters:
    df=input dataset
    shape=tensors shapes
    """

    dim_1, dim_2, _ = shape

    samples = df.shape[0]
    shape_matrix_2d = (samples, dim_1, dim_2, 2)
    #shape_1d_cons = (samples, dim_2, 1)  # 20nt

    x_sequence = df.iloc[:, 0].values.tolist()
    y_sequence = df.iloc[:, 1].values.tolist()
    z_cons = df.iloc[:, 2].values.tolist()

    ohe_matrix_2d = np.zeros(shape_matrix_2d)

    for sample in range(0, samples):
        z_cons_list = z_cons[sample].split(",")
        for x_seq_pos in range(0, dim_1):
            x_seq_nt = x_sequence[sample][x_seq_pos]
            for y_seq_pos in range(0, dim_2):
                y_seq_nt = y_sequence[sample][y_seq_pos]
                ohe_matrix_2d[sample, x_seq_pos, y_seq_pos, 0] = watson_crick(
                    x_seq_nt, y_seq_nt
                )
                ohe_matrix_2d[sample, x_seq_pos, y_seq_pos, 1] = z_cons_list[x_seq_pos]

    return ohe_matrix_2d


def make_sets_ohe(dataset, shape):
    """
    fun converts input batch into 
    one hot encoding of features
    and labels. output a tuple of 
    train test ohe dataframes and train test label dataframes.

    paramenters:
    batch=mini-batch as Pandas df
    """
    df_features = dataset.drop(["label"], axis=1)
    df_labels = dataset["label"]

    X_train = one_hot_encoding(df_features, shape)
    y_train = pd.get_dummies(df_labels).to_numpy()

    return X_train, y_train


def create_batches_points(samples, batches_size):
    """
    http://yaoyao.codes/pandas/2018/01/23/pandas-split-a-dataframe-into-chunks

    fun splits input pandas df into 
    batches. Returns a list of 
    subarrays.

    paramenters:
    samples=array of samles
    batches_size=chunk size
    """
    batches_points = list(
        range(
            1 * batches_size,
            (samples // batches_size + 1) * batches_size,
            batches_size,
        )
    )

    logging.info(
        f"split dataframe of shape {samples} into {len(batches_points)} +~ 1 mini-batches of size {batches_size}"
    )

    return batches_points


def load_dataset(target_tsv, shape):
    """
    fun loads connection table as pandas df,
    and return a list containing minibatches
    of connections as ohe (features) and labels.

    parameters:
    dataset=custard input tsv file
    """
    try:
        df = (
            pd.read_csv(target_tsv, sep="\t", names=["x", "y", "z", "label"])
            .sample(frac=1)
            .reset_index(drop=True)
        )
    except Exception as e:
        logging.error("Exception occured", exc_info=True)
        raise SystemExit("Failed to load dataset as pandas DataFrame")

    df_ohe, df_labels = make_sets_ohe(df, shape)
    
    return (df_ohe, df_labels)


if __name__ == "__main__":
    target_tsv = "pre_processing_test.tsv"
    load_dataset(target_tsv)
