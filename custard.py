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

    opt = OPTIONS["train"]

    dataset = pre_processing.load_dataset(
        infiles=opt["input_file"],
        tensor_dim=opt["tensor_dim"],
        read_file=opt["load_dataset"],
        save_datasets=opt["save_ohe"],
        output_dataset_filename=opt["output_dataset_filename"],
        train=OPTIONS["flags"]["train"],
        eval=OPTIONS["flags"]["evaluate"]
    )

    if OPTIONS["flags"]["train"]:
        model = train.do_training(
            opt, dataset
        )

    if OPTIONS["flags"]["evaluate"]:

        model = network.load_model_network(
            opt["model_output_dir"], opt["model_name"]
        )
        evaluate = network.model_evaluate(model, datasets, OPTIONS["evaluate"]["batch_size"])

    #     predictions = network.model_predict(model, dataset)
    #     evaluate.evaluate_model(
    #         y_true=labels,
    #         y_pred=predictions,
    #         threshold=OPTIONS["evaluate"]["threshold"],
    #         output_dir=OPTIONS["evaluate"]["model_dir"],
    #     )
