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
    # prepare dataset
    datasets = pre_processing.load_dataset(OPTIONS=OPTIONS)

    if OPTIONS["flags"]["train"]:
        model = train.do_training(
            
            OPTIONS, datasets
        
        )

    elif OPTIONS["flags"]["evaluate"]:

        model = network.load_model_network(
            
            OPTIONS["model_output_dir"], OPTIONS["model_name"]
        )
        evaluate = network.model_evaluate(model, datasets, OPTIONS["batch_size"])
    else:
        print("nothing to do, exit...")
        sys.exit()
    #     predictions = network.model_predict(model, dataset)
    #     evaluate.evaluate_model(
    #         y_true=labels,
    #         y_pred=predictions,
    #         threshold=OPTIONS["evaluate"]["threshold"],
    #         output_dir=OPTIONS["evaluate"]["model_dir"],
    #     )
