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
        alphabet = {"AT": 1, "TA": 1, "GC": 1, "CG": 1}
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
    logging.info(f'input table shape is {df.shape}')

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
                ohe_matrix[sample, x_seq_pos, y_seq_pos, 0] = watson_crick(
                    x_seq_nt, y_seq_nt
                )
    logging.info(f'output ohe shape is {ohe_matrix.shape}')
    return ohe_matrix


def load_dataset(
    target_tsv,
    scope='training'
    ):
    '''
    fun loads connection table as pandas df,
    and return 2 df of connections and labels.

    parameters:
    dataset=custard input tsv file
    '''
    try:
        df = pd.read_csv(
            target_tsv,
            sep='\t',
            names=['x','y','label']
            )
    except Exception as e:
        logging.error("Exception occured", exc_info=True)
        raise SystemExit
    df_connections = to_ohe( 
        df=df.drop(['label'],
        axis = 1
        ))

    df_labels = pd.get_dummies(df['label']).to_numpy()

    misc.load_dataset_checkpoint(
        df_connections,
        df_labels
    )

    if scope == 'training':
        return [(
            df_connections,
            df_labels
            )]
    elif scope in ['evaluation', 'predict']:
        return (
            df_connections,
            df_labels
            )
    else:
        logging.error(f'unknown scope for load dataset: {scope}')
        raise Exception(f'Unknown scope {scope}')


if __name__ == "__main__":
    target_tsv = "pre_processing_test.tsv"
    load_dataset(target_tsv)