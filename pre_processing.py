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


def to_ohe(df, dim_1, dim_2, dim_3=200):
    """
    fun transform input database to
    one hot encoding array.

    paramenters:
    df=input dataset
    """
    pos_dict = {"A": 0, "T": 1, "G": 2, "C": 3}

    samples = df.shape[0]
    shape_matrix_2d = (samples, dim_1, dim_2, 1)
    shape_cons_2d = (samples, dim_1, dim_3, 1)

    shape_1d_pirna = (samples, dim_2, 4)
    shape_1d_cons = (samples, dim_2, 1)  # 20nt
    shape_1d_bind = (samples, dim_1, 4)  # 50nt

    x_sequence = df.iloc[:, 0].values.tolist()
    y_sequence = df.iloc[:, 1].values.tolist()
    z_cons = df.iloc[:, 2].values.tolist()

    ohe_matrix_2d = np.zeros(shape_matrix_2d)
    ohe_cons_2d = np.zeros(shape_cons_2d)
    ohe_1d_bind = np.zeros(shape_1d_bind)
    ohe_1d_pirna = np.zeros(shape_1d_pirna)

    for sample in range(0, shape_matrix_2d[0]):
        z_cons_list = z_cons[sample].split(",")
        for x_seq_pos in range(0, shape_matrix_2d[1]):
            x_seq_nt = x_sequence[sample][x_seq_pos]
            nt_pos = pos_dict[x_seq_nt]
            ohe_1d_bind[sample, x_seq_pos, nt_pos] = 1
            for y_seq_pos in range(0, shape_matrix_2d[2]):
                y_seq_nt = y_sequence[sample][y_seq_pos]
                ohe_matrix_2d[sample, x_seq_pos, y_seq_pos, 0] = watson_crick(
                    x_seq_nt, y_seq_nt
                )
                nt_pos = pos_dict[y_seq_nt]
                ohe_1d_pirna[sample, y_seq_pos, nt_pos] = 1
            for z_cons_pos in range(0, shape_cons_2d[2]):
                ohe_cons_2d[sample, x_seq_pos, z_cons_pos, 0] = z_cons_list[z_cons_pos]

    return (samples, (ohe_matrix_2d, ohe_1d_bind, ohe_1d_pirna, ohe_cons_2d))


# def split_train_val_set(dataset_ohe, validation_split=0.5):
#     # batches for different model branches.
#     branch_1, branch_2, branch_3, batch_label = dataset_ohe

#     output_batches = []

#     for batch_features in [branch_1, branch_2, branch_3, batch_label]:
#         try:
#             X_train, X_test, y_train, y_test = train_test_split(
#                 batch_features,
#                 ,
#                 test_size=validation_split,
#                 random_state=1989,
#             )
#             output_batches.append([X_train, X_test, y_train, y_test])
#         except:
#             print(
#                 f"split_train_val_set - skip batch due to size:\t{batch_features.shape}"
#             )
#     return output_batches


def make_sets_ohe(dataset, dim_1, dim_2):
    """
    fun converts input batch into 
    one hot encoding of features
    and labels. output a tuple of 
    train test ohe dataframes and train test label dataframes.

    paramenters:
    batch=mini-batch as Pandas df
    """
    batch_features = dataset.drop(["label"], axis=1)
    batch_label = dataset["label"]

    samples, X_train_ohe = to_ohe(batch_features, dim_1, dim_2)

    y_train_dummies = pd.get_dummies(batch_label).to_numpy()

    return samples, X_train_ohe, y_train_dummies


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


def load_dataset(target_tsv, batch_size=32, dim_1=50, dim_2=20, scope="training"):
    """
    fun loads connection table as pandas df,
    and return a list containing minibatches
    of connections as ohe (features) and labels.

    parameters:
    dataset=custard input tsv file
    batch_size=split dataset into mini-batches
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

    samples, df_ohe, df_labels = make_sets_ohe(df, dim_1, dim_2)
    batches_points = create_batches_points(samples, batch_size)

    ohe_matrix_2d_batches = np.split(df_ohe[0], batches_points)
    ohe_matrix_1d_binds_batches = np.split(df_ohe[1], batches_points)
    ohe_matrix_1d_pirna_batches = np.split(df_ohe[2], batches_points)
    ohe_cons_2d_batches = np.split(df_ohe[3], batches_points)
    df_labels_batches = np.split(df_labels, batches_points)

    mini_batches_set = list()
    for number, batch in enumerate(
        zip(
            ohe_matrix_2d_batches,
            ohe_matrix_1d_binds_batches,
            ohe_matrix_1d_pirna_batches,
            ohe_cons_2d_batches,
            df_labels_batches,
        )
    ):

        dataset = batch
        # if scope == "training":
        #     dataset = split_train_val_set(batch)
        # else:
        #     dataset = batch
        mini_batches_set.append(dataset)
    return mini_batches_set


if __name__ == "__main__":
    target_tsv = "pre_processing_test.tsv"
    load_dataset(target_tsv)
