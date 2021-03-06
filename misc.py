import logging
import pprint
import os
import json
import sys


def load_options():

    with open("/home/grioni_andrea/loft/custard/AG_custard_opt.json", "r") as fp:
        OPTIONS = json.load(fp)
    # try:
    #     with open(sys.argv[1], "r") as fp:
    #         OPTIONS = json.load(fp)
    # except IndexError:
    #     OPTIONS = json.load)
    #     # print("please provide input confing json file, exit...")
    #     # sys.exit()

    create_log(OPTIONS)
    print_options_log(OPTIONS)

    return OPTIONS


def create_log(OPTIONS):
    """
    fun creates logging file
    
    paramenters:
    OPTIONS=tool arguments
    """
    # input_paramenters_checkpoint(OPTIONS)

    level = OPTIONS["log"]["level"]
    file_name = OPTIONS["log"]["name"]
    # FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
    logging.basicConfig(
        # format=FORMAT,
        filemode="w",
        level=logging.DEBUG,
        filename=file_name,
    )
    return None


def print_options_log(OPTIONS):
    """
    fun writes to logging file 
    the tool arguments.

    paramenters:
    options=tool arguments
    """
    log_string = pprint.pformat(OPTIONS, indent=1, width=80, depth=None, compact=True)
    logging.info(log_string)
    return None


def load_dataset_checkpoint(number, batch_shape, batch_ohe):
    """
    checkpoint function for module
    pre_processing - load_dataset.
    Assert that dataframe splitting 
    occurred without errors.

    paramenters:
    number=batch number
    batch_shape=shape of input batch
    batch_ohe=tuple of transformed batch to ohe
    """
    if batch_ohe[0].shape[0] != batch_ohe[1].shape[0]:
        error_msg = f"different train-label samples shapes:\t{batch_ohe[0].shape} != {batch_ohe[1].shape}"
        logging.error(error_msg)
        raise Exception(error_msg)
        raise SystemExit

    logging.info(
        f"batch\t{number}\tbatch_shape\t{batch_shape}\ttrain-ohe\t{batch_ohe[0].shape}\tlabel-ohe\t{batch_ohe[1].shape}"
    )
    return None


def input_paramenters_checkpoint(OPTIONS):
    """
    fun controls that tool arguments
    are correct.

    paramenters:
    OPTIONS=tool arguments
    """
    if not os.path.exists(OPTIONS["input_file"]):
        logging.error(f'input file does not exit: {OPTIONS["input_file"]}')
        raise FileNotFoundError(OPTIONS["input_file"])
        raise SystemExit
    return None


def print_history(
    iteration=None,
    batch=None,
    train_set_size=None,
    batch_limit=None,
    train_batch_size=None,
    test_batch_size=None,
    train_batch_history=None,
    test_batch_history=None,
    history=None,
    epoch_train=True,
):
    if epoch_train:
        format_string = f"\titer\t{iteration}\tbatch\t{batch}|{train_set_size}\ttrain_size|{train_batch_size}\tloss|{train_batch_history[0]:.2E}\taccuracy|{train_batch_history[1]:.2f}\tval_size|{test_batch_size}\tloss|{test_batch_history[0]:.2E}\taccuracy|{test_batch_history[1]:.2f}"

        print(format_string)
        logging.info(format_string)

        log_history = f"{iteration}\t{batch}\t{train_batch_size}\t{train_batch_history[0]}\t{train_batch_history[1]}\t{test_batch_size}\t{test_batch_history[0]}\t{test_batch_history[1]}"
    else:
        log_history = f'\t{iteration}\t{batch}\t{train_set_size}\t{test_batch_size}\t{history["accuracy"]:.2f}\t{history["loss"]:.2E}\t{history["val_accuracy"]:.2f}\t{history["val_loss"]:.2E}'

    return log_history


def update_history(history, log_history, iteration=None):
    """
    update dataframe with training
    history.

    parameters:
    history=df with current history
    log_history=df with previous histories
    iteration=iteration number
    """
    df_history = pd.DataFrame(history)
    if iteration:
        df_history["iteration"] = iteration
    updated_history = log_history.append(df_history)
    return updated_history


if __name__ == "__main__":
    pass


def write_table(evaluate, OPTIONS):
    loss, val = evaluate[0], evaluate[1]
    string_out = "loss\tacc\n%s\t%s\n" % (loss, val)
    with open(OPTIONS["evaluate"]["eval_file_output"], "w") as fo:
        fo.write(string_out)
    return OPTIONS["evaluate"]["eval_file_output"]
