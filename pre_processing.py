from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import misc
import logging

'''
this module creates input dataset
for custard
'''


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
        alphabet = {
            "AT": 1,
            "TA": 1,
            "GC": 1,
            "CG": 1
            }
    pair = x_nt + y_nt
    return alphabet.get(pair, 0)


def to_ohe(
    df
    ):
    '''
    fun transform input database to
    one hot encoding array.

    paramenters:
    df=input dataset
    '''
    samples = df.shape[0]
    ohe_shape = (samples, 50, 20, 1)

    x_sequence = df.iloc[ : , 0].values.tolist()
    y_sequence = df.iloc[ : , 1].values.tolist()

    ohe_matrix = np.zeros(ohe_shape)

    for sample in range(0, ohe_shape[0]):
        for x_seq_pos in range(0, ohe_shape[1]):
            x_seq_nt = x_sequence[sample][x_seq_pos]
            for y_seq_pos in range(0, ohe_shape[2]):
                y_seq_nt = y_sequence[sample][y_seq_pos]
                ohe_matrix[
                    sample,
                    x_seq_pos,
                    y_seq_pos,
                    0] = watson_crick(
                    x_seq_nt, y_seq_nt
                )
    return ohe_matrix

def split_train_val_set(
    dataset_ohe, validation_split = 0.2
    ):
    batch_features, batch_label = dataset_ohe
    X_train, X_test, y_train, y_test = train_test_split(
        batch_features, batch_label,
        test_size=validation_split,
        random_state=1989)
    return (X_train, X_test,
        y_train, y_test)


def make_sets_ohe(
    dataset
    ):
    '''
    fun converts input batch into 
    one hot encoding of features
    and labels. output a tuple of 
    train test ohe dataframes and train test label dataframes.

    paramenters:
    batch=mini-batch as Pandas df
    '''
    batch_features = dataset.drop(
        ['label'],
        axis = 1
    )
    batch_label = dataset['label']

    X_train_ohe = to_ohe(batch_features)

    y_train_dummies = pd.get_dummies(
        batch_label ).to_numpy()

    return ( X_train_ohe, y_train_dummies )


def load_dataset(
    target_tsv,
    batch_size,
    scope='training'
    ):
    '''
    fun loads connection table as pandas df,
    and return 2 df of connections and labels.

    parameters:
    dataset=custard input tsv file
    batch_size=split dataset into mini-batches
    '''
    try:
        df = pd.read_csv(
            target_tsv,
            sep='\t',
            names=['x','y','label']
            ).sample(frac=1).reset_index(drop=True)
    except Exception as e:
        logging.error("Exception occured", exc_info=True)
        raise SystemExit("Failed to load dataset as pandas DataFrame")
    
    df_ohe = make_sets_ohe(df)
    df_ohe_batches, df_ohe_labels = split_df(df_ohe, batch_size)

    if scope == 'training':
        train_set = list()
        for number, batch in enumerate(
            zip(df_ohe_batches, df_ohe_labels)
            ):
            dataset = split_train_val_set(batch)

            train_set.append(
                dataset
            )

        return train_set

    elif scope in ['evaluation', 'predict']:
        return (
            df_connections,
            df_labels
            )
    else:
        logging.error(f'unknown scope for load dataset: {scope}')
        raise Exception(f'Unknown scope {scope}')

# author: http://yaoyao.codes/pandas/2018/01/23/pandas-split-a-dataframe-into-chunks
def chunk_marks(nrows, chunk_size):
    '''
    fun generates a 1D array that indicate
    where to split the df
    
    paramenters:
    nrows=df shape
    chunk_size=batch size
    '''
    split_arrays = range(1 * chunk_size,
    (nrows // chunk_size + 1) * chunk_size, chunk_size
    )
    return split_arrays

def split_df(df, batches_size):
    '''
    fun splits input pandas df into 
    batches. Returns a list of 
    subarrays.

    paramenters:
    df=input pandas df
    batches_size=chunk size
    '''
    df_ohe, df_labels = df
    batches_points = list(
        chunk_marks(df_ohe.shape[0], batches_size)
        )
    df_ohe_batches = np.split(df_ohe, batches_points)
    df_ohe_labels = np.split(df_labels, batches_points)
    
    logging.info(f"split dataframe of shape {df_ohe.shape} into {len(batches_points)} +~ 1 mini-batches of size {batches_size}")
    
    return df_ohe_batches, df_ohe_labels

if __name__ == "__main__":
    target_tsv = "pre_processing_test.tsv"
    load_dataset(target_tsv)