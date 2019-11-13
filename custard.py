import os
import network
import train
import pre_processing
import predict
import evaluate
import misc

def data_gen(target_file, scope='training'):
    train_set = pre_processing.load_dataset(
        target_tsv=target_file,
        scope=scope
    )
    return train_set

def do_training(OPTIONS):
    train_opt = OPTIONS['train']
    # load dataset
    input_file = OPTIONS['input_file']
    dataset = data_gen(input_file)
    # generate network
    model = network.build_network()
    # train network
    history, model = train.train_network(
        model,
        dataset,
        batches_limit=train_opt[
            'batches_limit'
            ],
        iterations=train_opt['iterations']
    )
    # save model
    network.save_model(model, os.getcwd())
    return history, model


if __name__ == "__main__":

    OPTIONS = {
        'flags' : {
            'train' : True,
            'predict' : False,
            'evaluate' : False
        },
        'threshold' : 0.5,
        'input_file' : '/home/angri/Desktop/projects/custard_testing/custard_toy.tsv',
        'model' : {
            'name' : None,
            'path' : None
        },
        'log' : {
            'level' : 'debug',
            'name' : 'test_logging.txt'
        },
        'train' : {
            'iterations' : 10,
            'epochs' : 10,
            'batches_limit' : 10,
            'neg_increments' : False
        }
    }

    misc.create_log(OPTIONS)
    misc.options_log(OPTIONS)

    
    model_name = OPTIONS['model']['name']
    model_path = OPTIONS['model']['path']

    if OPTIONS['flags']['train']:
        history, model = do_training(OPTIONS)

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