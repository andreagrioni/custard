import logging
import pprint
import os

def create_log(OPTIONS):
    '''
    fun creates logging file
    
    paramenters:
    OPTIONS=tool arguments
    '''
    level = OPTIONS['log']['level']
    file_name = OPTIONS['log']['name']
    #FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
    logging.basicConfig(
        #format=FORMAT,
        filemode='w',
        level=logging.DEBUG,
        filename=file_name)
    
    input_paramenters_checkpoint(OPTIONS)
    return None


def options_log(OPTIONS):
    '''
    fun writes to logging file 
    the tool arguments.

    paramenters:
    options=tool arguments
    '''
    log_string = pprint.pformat(
        OPTIONS,
        indent=1,
        width=80,
        depth=None,
        compact=True
        )
    logging.info(log_string)
    return None


def load_dataset_checkpoint(df_connections, df_labels):
    '''
    checkpoint function for module
    pre_processing - load_dataset.
    Assert that dataframe splitting 
    occurred without errors.

    paramenters:
    df_connections=train df
    df_labels=labels
    '''
    if df_connections.shape[0] != df_labels.shape[0]:
        error_msg = f'different train-label samples shapes:\t{df_connections.shape} != {df_labels.shape}'
        logging.error(
            error_msg
            )
        raise Exception(error_msg)
    return None


def input_paramenters_checkpoint(
    OPTIONS
    ):
    '''
    fun controls that tool arguments
    are correct.

    paramenters:
    OPTIONS=tool arguments
    '''
    if not os.path.exists(OPTIONS['input_file']):
        logging.error(f'input file does not exit: {OPTIONS["input_file"]}')
        raise FileNotFoundError(OPTIONS['input_file'])
        raise SystemExit
    if OPTIONS['model']['path'] and OPTIONS['model']['name']:
        model_path = os.path.join(
            OPTIONS['model']['path'],
            OPTIONS['model']['name']
            )
        if not os.path.exists(
            OPTIONS['model']['path']
            ):
            logging.error(
    f'model dir does not exit: {OPTIONS["model"]["path"]}'
                )
            raise FileNotFoundError(
        OPTIONS['model']['path']
        )
            raise SystemExit
        
        if not os.path.exists(model_path):
            logging.error(
    f'model file does not exit: {model_path}')
            raise FileNotFoundError(
        model_path
        )
            raise SystemExit