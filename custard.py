import os
import network
import train
import pre_processing
import predict
import evaluate_metrics
import misc
import sys


def data_gen(target_file, shape):
    converted_datasets = pre_processing.load_dataset(
        target_tsv=target_file,
        shape=shape
    )
    return converted_datasets


def do_training(OPTIONS):
    train_opt = OPTIONS["train"]
    # train settings
    batch_size = train_opt["batch_size"]
    
    training_table = OPTIONS["input_file"]
    validation_table = train_opt["val_dataset"]
    
    classes = train_opt["classes"]
    shape = (
        train_opt["dim_1"],
        train_opt["dim_2"],
        train_opt["dim_3"]
        )
    # load train dataset
    train_dataset = data_gen(
        training_table,
        shape
    )
    # load validation dataset
    val_dataset = data_gen(
        validation_table,
        shape
    )
    # generate network
    model = network.build_network(
        classes=classes, shape=shape
        )
    # train network
    model = train.train_network(
        model,
        train_dataset,
        val_dataset,
        batch_size=batch_size
    )
    # save model
    network.save_model(model, os.getcwd())
    return model


# def do_prediction(OPTIONS, dataset):
#     eval_opt = OPTIONS["evaluate"]
#     batch_size = eval_opt["batch_size"]
#     input_file = OPTIONS["input_file"]

#     # load model
#     model_file_path = os.path.join(eval_opt["model_dir"], eval_opt["model"])
#     model = network.load_model_network(model_file_path)

#     # make predictions
#     predictions = predict.model_predict(model, dataset, batch_size=batch_size)

#     return predictions


if __name__ == "__main__":
    try:
        json_filepath = sys.argv[1]
    except:
        json_filepath = None

    OPTIONS = misc.load_options(json_filepath)

    if OPTIONS["flags"]["train"]:
        model = do_training(OPTIONS)

    # if OPTIONS["flags"]["evaluate"] or OPTIONS["flags"]["predict"]:
    #     # load dataset
    #     eval_opt = OPTIONS["evaluate"]
    #     batch_size = eval_opt["batch_size"]
    #     input_file = OPTIONS["input_file"]
    #     metrics_filename = eval_opt["metrics_filename"]
    #     dataset = data_gen(input_file, batch_size, scope="evaluate")

    #     predictions = do_prediction(OPTIONS, dataset)

    # if OPTIONS["flags"]["evaluate"]:

    #     evaluate_metrics.evaluate(predictions, json_name=metrics_filename)

