# import network
import train
import pre_processing
import predict
import evaluate_metrics
import evaluate
import misc
import sys
import os
import network

if __name__ == "__main__":

    OPTIONS = misc.load_options()

    if OPTIONS["flags"]["train"]:
        dataset = pre_processing.load_dataset(
            dataset=OPTIONS["train"]["input_file"],
            tensor_dim=OPTIONS["train"]["tensor_dim"],
        )
        model = train.do_training(
            OPTIONS, dataset, tensor_dim=OPTIONS["train"]["tensor_dim"]
        )
    if OPTIONS["flags"]["evaluate"]:
        dataset, labels = pre_processing.load_dataset(
            dataset=OPTIONS["evaluate"]["input_file"],
            tensor_dim=OPTIONS["evaluate"]["tensor_dim"],
        )
        model = network.load_model_network(
            OPTIONS["evaluate"]["model"], OPTIONS["evaluate"]["model_dir"]
        )
        predictions = network.model_predict(model, dataset)
        evaluate.evaluate_model(
            y_true=labels,
            y_pred=predictions,
            threshold=OPTIONS["evaluate"]["threshold"],
            output_dir=OPTIONS["evaluate"]["model_dir"],
        )
