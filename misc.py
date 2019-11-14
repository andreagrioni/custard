import logging
import pprint
import os


def load_options():
    OPTIONS = {
        'flags' : {
            'train' : True,
            'predict' : False,
            'evaluate' : False
        },
        'threshold' : 0.5,
        'input_file' : '/home/angri/Desktop/projects/custard_testing/custard_toy.tsv',
        'working_dir' : '/home/angri/Desktop/projects/custard_testing/',
        'model' : {
            'name' : None,
            'path' : None
        },
        'log' : {
            'level' : 'debug',
            'name' : 'test_logging.txt'
        },
        'train' : {
            'iterations' : 5,
            'epochs' : 10,
            'batch_size' : 32,
            'batches_limit' : 30,
            'neg_increments' : False,
            'classes' : 3
        }
    }

    create_log(OPTIONS)
    print_options_log(OPTIONS)

    return OPTIONS


def create_log(OPTIONS):
    '''
    fun creates logging file
    
    paramenters:
    OPTIONS=tool arguments
    '''
    input_paramenters_checkpoint(OPTIONS)
    os.chdir(OPTIONS['working_dir'])
    level = OPTIONS['log']['level']
    file_name = OPTIONS['log']['name']
    #FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
    logging.basicConfig(
        #format=FORMAT,
        filemode='w',
        level=logging.DEBUG,
        filename=file_name)
    logging.info(
        f'change wd at: {OPTIONS["working_dir"]}'
        )
    return None


def print_options_log(OPTIONS):
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


def load_dataset_checkpoint(
    number, batch_shape,
    batch_ohe
    ):
    '''
    checkpoint function for module
    pre_processing - load_dataset.
    Assert that dataframe splitting 
    occurred without errors.

    paramenters:
    number=batch number
    batch_shape=shape of input batch
    batch_ohe=tuple of transformed batch to ohe
    '''
    if batch_ohe[0].shape[0] != batch_ohe[1].shape[0]:
        error_msg = f'different train-label samples shapes:\t{df_connections.shape} != {df_labels.shape}'
        logging.error(
            error_msg
            )
        raise Exception(error_msg)
        raise SystemExit

    logging.info(f'batch\t{number}\tbatch_shape\t{batch_shape}\ttrain-ohe\t{batch_ohe[0].shape}\tlabel-ohe\t{batch_ohe[1].shape}')

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
    if not os.path.exists(OPTIONS["working_dir"]):
        os.makedirs(
            OPTIONS["working_dir"]
            )
        logging.info(
            f'create wd: {OPTIONS["working_dir"]}'
            )
    return None


def print_history(
    iteration,
    batch,
    train_set_size,
    batch_limit,
    train_batch_size,
    test_batch_size,
    train_batch_history,
    test_batch_history
    ):

    format_string = (f'\titer\t{iteration}\tbatch\t{batch}|{train_set_size}\ttrain_size|{train_batch_size}\tloss|{train_batch_history[0]:.2E}\taccuracy|{train_batch_history[1]:.2f}\tval_size|{test_batch_size}\tloss|{test_batch_history[0]:.2E}\taccuracy|{test_batch_history[1]:.2f}')

    print(format_string)
    logging.info(format_string)

    log_history = f'{iteration}\t{batch}\t{train_batch_size}\t{train_batch_history[0]}\t{train_batch_history[1]}\t{test_batch_size}\t{test_batch_history[0]}\t{test_batch_history[1]}'

    return log_history

def update_history(
    history,
    log_history,
    iteration=None
    ):
    '''
    update dataframe with training
    history.

    parameters:
    history=df with current history
    log_history=df with previous histories
    iteration=iteration number
    '''
    df_history = pd.DataFrame(
        history
        )
    if iteration:
        df_history['iteration'] = iteration
    updated_history = log_history.append(
        df_history
        )
    return updated_history

if __name__ == "__main__":
    pass