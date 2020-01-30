from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import misc
import logging
import os
import time

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
    if tensor_dim[-1] == 1:
        multichannel = False
    elif tensor_dim[-1] == 2:
        multichannel = 2
    elif tensor_dim[-1] == 3:
        multichannel = 3
    start = time.time()

    for index, row in df.iterrows():
        if multichannel:
            sample_bind_score = list(map(float, row.binding_cons_score.split(",")))
            sample_mirna_score = list(map(float, row.mirna_cons_score.split(",")))

        for bind_index, bind_nt in enumerate(row.binding_sequence):
            if multichannel:
                nt_bind_cons_score = sample_bind_score[bind_index]

            for mirna_index, mirna_nt in enumerate(row.mirna_binding_sequence):

                ohe_matrix_2d[index, bind_index, mirna_index, 0] = watson_crick(
                    bind_nt, mirna_nt
                )
                if multichannel == 2:
                    cons_score = nt_bind_cons_score * sample_mirna_score[mirna_index]
                    ohe_matrix_2d[index, bind_index, mirna_index, 1] = cons_score
                elif multichannel == 3:
                    ohe_matrix_2d[index, bind_index, mirna_index, 1] = nt_bind_cons_score
                    ohe_matrix_2d[index, bind_index, mirna_index, 2] = sample_mirna_score[mirna_index]
        if index % 1000 == 0:
            end = time.time()
            print(

                "rows:\t%s" %(index),
                "elapsed (sec):\t%s" %(end - start),
                "multichannel:\t%s" %(multichannel),
                sep=" | ",
            )
            ## debug
            # if index == 0:
            #     nt_pair = watson_crick(bind_nt, mirna_nt)
            #     print(row.binding_sequence)
            #     print(row.mirna_binding_sequence)
            #     print(bind_nt, mirna_nt, nt_pair)
            #     print(
            #         sample_bind_score[bind_index],
            #         sample_mirna_score[mirna_index],
            #         cons_score,
            #     )
            # break
    return ohe_matrix_2d


def make_sets_ohe(dataset, tensor_dim):
    """
    fun converts input batch into 
    one hot encoding of features
    and labels. output a tuple of 
    train test ohe dataframes and train test label dataframes.

    paramenters:
    batch=mini-batch as Pandas df
    """
    df_features = dataset.drop(["label"], axis=1)

    X_train = one_hot_encoding(df_features, tensor_dim)
    y_train = pd.get_dummies(dataset.label).to_numpy()

    return [X_train, y_train]


def load_dataset(
    infiles,
    tensor_dim=None,
    read_file=False,
    save_datasets=False,
    output_dataset_filename="custard.ohe.npz",
    train=False,
    eval=False

):
    """
    fun loads connection table as pandas df,
    and return a list containing minibatches
    of connections as ohe (features) and labels.

    parameters:
    infiles=tuple of train,validation or test set
    tensor_dim=tensor dimensions
    scope=model stage (train, validation, pred)
    read_file=load datasets from file
    """

    if read_file:
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
                datasets = [ data["X_test"], data["y_test"] ]
    else:
        print("converting files to ohe datasets:", infiles, sep="\t")
        datasets = []
        for target_tsv in infiles:
            if target_tsv:
                try:
                    df = (
                        pd.read_csv(
                            target_tsv, sep="\t"
                        )  # , names=["x", "y", "z", "label"])
                        .sample(frac=1)
                        .reset_index(drop=True)
                    )
                except Exception as e:
                    logging.error("Exception occured", exc_info=True)
                    raise SystemExit("Failed to load dataset as pandas DataFrame")
                data_ohe = make_sets_ohe(df, tensor_dim)
                datasets += data_ohe
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
                    output_dataset_filename,
                    X_test=datasets[0],
                    y_test=datasets[1],
                #    y_train=datasets[1],
                #    y_val=datasets[3],    
                )
    return datasets


if __name__ == "__main__":
    load_dataset(["toy/toy_train.tsv", "toy/toy_val.tsv"], (200, 20, 2))
    pass
