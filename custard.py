import os
import network
import train
import pre_processing
import predict
import evaluate_metrics
import misc
import sys


def data_gen(target_file, batch_size, dim_1, dim_2, scope="training"):
    train_set = pre_processing.load_dataset(
        target_tsv=target_file,
        batch_size=batch_size,
        dim_1=dim_1,
        dim_2=dim_2,
        scope=scope,
    )
    return train_set


def do_training(OPTIONS):
    train_opt = OPTIONS["train"]
    # train settings
    batch_size = train_opt["batch_size"]
    batches_limit = train_opt["batches_limit"]
    iterations = train_opt["iterations"]
    input_file = OPTIONS["input_file"]
    classes = train_opt["classes"]
    dim_1 = train_opt["dim_1"]
    dim_2 = train_opt["dim_2"]
    # load dataset
    dataset = data_gen(input_file, batch_size, dim_1, dim_2)
    # generate network
    model = network.build_network(classes, dim_1, dim_2)
    # train network
    model = train.train_network(
        model, dataset, batches_limit=batches_limit, iterations=iterations
    )
    # save model
    network.save_model(model, os.getcwd())
    return model


def do_prediction(OPTIONS, dataset):
    eval_opt = OPTIONS["evaluate"]
    batch_size = eval_opt["batch_size"]
    input_file = OPTIONS["input_file"]

    # load model
    model_file_path = os.path.join(eval_opt["model_dir"], eval_opt["model"])
    model = network.load_model_network(model_file_path)

    # make predictions
    predictions = predict.model_predict(model, dataset, batch_size=batch_size)

    return predictions


if __name__ == "__main__":
    try:
        json_filepath = sys.argv[1]
    except:
        json_filepath = None
    OPTIONS = misc.load_options(json_filepath)

    if OPTIONS["flags"]["train"]:
        model = do_training(OPTIONS)

    if OPTIONS["flags"]["evaluate"] or OPTIONS["flags"]["predict"]:
        # load dataset
        eval_opt = OPTIONS["evaluate"]
        batch_size = eval_opt["batch_size"]
        input_file = OPTIONS["input_file"]
        metrics_filename = eval_opt["metrics_filename"]
        dataset = data_gen(input_file, batch_size, scope="evaluate")

        predictions = do_prediction(OPTIONS, dataset)

    if OPTIONS["flags"]["evaluate"]:

        evaluate_metrics.evaluate(predictions, json_name=metrics_filename)

