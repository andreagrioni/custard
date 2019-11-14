import os
import network
import train
import pre_processing
import predict
import evaluate
import misc

def data_gen(
    target_file, batch_size, scope='training'
    ):
    train_set = pre_processing.load_dataset(
        target_tsv=target_file,
        batch_size=batch_size,
        scope=scope,
    )
    return train_set

def do_training(OPTIONS):
    train_opt = OPTIONS['train']
    # train settings
    batch_size = train_opt['batch_size']
    batches_limit = train_opt['batches_limit']
    iterations = train_opt['iterations']
    input_file = OPTIONS['input_file']
    classes = train_opt['classes']
    # load dataset
    dataset = data_gen(input_file, batch_size)
    # generate network
    model = network.build_network(classes)
    # train network
    model = train.train_network(
        model,
        dataset,
        batches_limit=batches_limit,
        iterations=iterations
    )
    # save model
    network.save_model(model, os.getcwd())
    return model


if __name__ == "__main__":

    OPTIONS = misc.load_options()    
    
    #model_name = OPTIONS['model']['name']
    #model_path = OPTIONS['model']['path']    

    if OPTIONS['flags']['train']:
        model = do_training(
            OPTIONS
            )

    if OPTIONS['flags']['predict']:
        X_true, y_true = data_gen(
            input_file, scope='predict'
            )
        model = predict.load(
            model_path, model_name
            )
        y_pred = predict.model_predict(
            model,
            X_true
        )

    if OPTIONS['flags']['evaluate']:
        y_pred_pos = y_pred[ : , 1]
        y_true_pos = y_true[: , 1]
        evaluate.evaluate_model(
            y_true_pos,
            y_pred_pos,
            threshold,
            output_dir=None,
            json_name='metrics'
        )