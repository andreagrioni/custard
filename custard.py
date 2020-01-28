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
    dataset = pre_processing.load_dataset(
        infiles=OPTIONS["train"]["input_file"],
        tensor_dim=OPTIONS["train"]["tensor_dim"],
        read_file=OPTIONS["train"]["load_dataset"],
        save_datasets=OPTIONS["train"]["save_ohe"],
        output_dataset_filename=OPTIONS["train"]["output_dataset_filename"],
    )

    if OPTIONS["flags"]["train"]:
        model = train.do_training(
            OPTIONS, dataset
        )

    # if OPTIONS["flags"]["evaluate"]:
    #     dataset, labels = pre_processing.load_dataset(
    #         dataset=OPTIONS["evaluate"]["input_file"],
    #         tensor_dim=OPTIONS["evaluate"]["tensor_dim"],
    #     )
    #     model = network.load_model_network(
    #         OPTIONS["evaluate"]["model"], OPTIONS["evaluate"]["model_dir"]
    #     )
    #     predictions = network.model_predict(model, dataset)
    #     evaluate.evaluate_model(
    #         y_true=labels,
    #         y_pred=predictions,
    #         threshold=OPTIONS["evaluate"]["threshold"],
    #         output_dir=OPTIONS["evaluate"]["model_dir"],
    #     )
