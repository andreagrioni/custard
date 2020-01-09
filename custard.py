# import network
import train
import pre_processing
import predict
import evaluate_metrics
import misc
import sys
import os

if __name__ == "__main__":

    OPTIONS = misc.load_options()

    if OPTIONS["flags"]["train"]:
        dataset = pre_processing.load_dataset(
            train_tsv=OPTIONS["input_file"],
            validation_tsv=OPTIONS['train']['val_dataset'],
            tensor_dim=OPTIONS['train']["tensor_dim"]
        )
        model = train.do_training(
            OPTIONS, dataset, tensor_dim=OPTIONS['train']["tensor_dim"]
            )
    
