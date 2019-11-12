import os
import network
import train
import pre_processing
import predict
import evaluate

def data_gen(target_file, scope='training'):
    train_set = pre_processing.load_dataset(
        target_tsv=target_file,
        scope=scope
    )
    return train_set

def do_training(dataset):
    # generate network
    model = network.build_network()

    # train network
    history, model = train.train_network(
        model,
        dataset,
        batches_limit=None,
        iterations=None
    )
    return history, model


if __name__ == "__main__":
    train_flag = False
    predict_flag = True
    evaluate_flag = True
    threshold = 0.5
    target_file = 'pre_processing_test.tsv'
    model_path = '/home/angri/Desktop/projects/custard/'
    model_name= 'my_model.h5'

    if train_flag:
        dataset = data_gen(target_file)
        history, model = do_training(dataset)
        network.save_model(model, os.getcwd())
    if predict_flag:
        X_true, y_true = data_gen(
            target_file, scope='predict'
            )
        model = predict.load(
            model_path, model_name
            )
        y_pred = predict.model_predict(
            model,
            X_true
        )
    if evaluate_flag:
        y_pred_pos = y_pred[ : , 1]
        y_true_pos = y_true[: , 1]
        evaluate.evaluate_model(
            y_true_pos, y_pred_pos, threshold,
            output_dir=None, json_name='metrics'
        )