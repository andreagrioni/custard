from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import time
import sys
import json

"""
convert table to one hot encoding, save as numpy array.
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


def one_hot_encoding(df, tensor_dim):
    """
    fun transform input database to
    one hot encoding array.

    paramenters:
    df=input dataset
    tensor_dim=tensors shapes
    """

    samples = df.shape[0]
    # matrix of 4D with samples, nucleotide
    # binding length, miRNA length,
    # channels (dot matrix and conservation)
    shape_matrix_2d = (samples, *tensor_dim)
    ohe_matrix_2d = np.zeros(shape_matrix_2d, dtype="float32")
    multichannel = tensor_dim[-1]

    start = time.time()

    for index, row in df.iterrows():
        if multichannel > 1:
            sample_bind_score = list(map(float, row.binding_cons_score.split(",")))
            sample_mirna_score = list(map(float, row.mirna_cons_score.split(",")))

        for bind_index, bind_nt in enumerate(row.binding_sequence):
            if multichannel > 1:
                nt_bind_cons_score = sample_bind_score[bind_index]

            for mirna_index, mirna_nt in enumerate(row.mirna_binding_sequence):

                ohe_matrix_2d[index, bind_index, mirna_index, 0] = watson_crick(
                    bind_nt, mirna_nt
                )
                if multichannel == 2:
                    cons_score = nt_bind_cons_score * sample_mirna_score[mirna_index]
                    ohe_matrix_2d[index, bind_index, mirna_index, 1] = cons_score
                elif multichannel == 3:
                    ohe_matrix_2d[
                        index, bind_index, mirna_index, 1
                    ] = nt_bind_cons_score
                    ohe_matrix_2d[
                        index, bind_index, mirna_index, 2
                    ] = sample_mirna_score[mirna_index]
        if index % 1000 == 0:
            end = time.time()
            print(
                "rows:\t%s" % (index),
                "elapsed (sec):\t%s" % (end - start),
                "multichannel:\t%s" % (multichannel),
                sep=" | ",
            )
    return ohe_matrix_2d


def make_sets_ohe(samples, labels, tensor_dim):
    """
    fun converts input batch into 
    one hot encoding of features
    and labels. output a tuple of 
    train test ohe dataframes and train test label dataframes.

    paramenters:
    batch=mini-batch as Pandas df
    """
    X = samples.reset_index(drop=True).copy()
    y = labels.reset_index(drop=True).copy()

    X_ohe = one_hot_encoding(X, tensor_dim)
    y_ohe = pd.get_dummies(y).to_numpy()

    return X_ohe, y_ohe


def load_dataset(OPTIONS, main=False):
    """
    fun loads connection table as pandas df to
    one hot encoding array.

    parameters:
    OPTIONS=input custard options (dict)
    """

    infiles = OPTIONS["input_file"]
    tensor_dim = OPTIONS["tensor_dim"]
    restore_dataset = OPTIONS["load_dataset"]
    save_datasets = OPTIONS["save_ohe"]
    output_dataset_filename = OPTIONS["output_ohe_datasets_name"]
    make_validation = OPTIONS["validation"]
    train = OPTIONS["flags"]["train"]
    eval = OPTIONS["flags"]["evaluate"]

    if eval:
        datasets = [0, 0]
    elif train:
        datasets = [0, 0, 0, 0]
    elif main:
        datasets = [0, 0, 0, 0]
        print("running as stand-alone script")
    else:
        print("nothing to do, exit...")
        sys.exit()

    if restore_dataset:
        print("load dataset from file:", infiles, sep="\t")
        if train:
            with np.load(infiles) as data:
                datasets = [
                    data["X_train"],
                    data["y_train"],
                    data["X_val"],
                    data["y_val"],
                ]
        elif eval:
            with np.load(infiles) as data:
                datasets = [data["X_test"], data["y_test"]]
    else:
        print("converting files to ohe datasets:", infiles, sep="\t")

        try:
            df = pd.read_csv(infiles, sep="\t")
        except Exception as e:
            if not main:
                logging.error("Exception occured", exc_info=True)
            raise SystemExit("Failed to load dataset as pandas DataFrame")

        if make_validation:
            y_samples = df.label
            X_samples = df.drop(["label"], axis=1)
            X_train, X_val, y_train, y_val = train_test_split(
                X_samples, y_samples, test_size=0.2, random_state=1989
            )
        else:
            print("training with no validation is not implemented yet, exit...")
            sys.exit()

        datasets[0], datasets[1] = make_sets_ohe(X_train, y_train, tensor_dim)
        datasets[2], datasets[3] = make_sets_ohe(X_val, y_val, tensor_dim)

        if save_datasets:
            print("saving ohe datasets at location:", output_dataset_filename, sep="\t")
            if len(datasets) == 4:
                np.savez(
                    output_dataset_filename,
                    X_train=datasets[0],
                    X_val=datasets[2],
                    y_train=datasets[1],
                    y_val=datasets[3],
                )
            elif len(datasets) == 2:
                np.savez(
                    output_dataset_filename, X_test=datasets[0], y_test=datasets[1],
                )
    return datasets


if __name__ == "__main__":
    try:
        with open(sys.argv[1], "r") as fp:
            OPTIONS = json.load(fp)
    except:
        with open(
            "/home/grioni_andrea/loft/custard/AG_table2ohe_confing.json", "r"
        ) as fp:
            OPTIONS = json.load(fp)

    load_dataset(OPTIONS, main=True)
