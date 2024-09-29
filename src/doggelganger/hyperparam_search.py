import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from torch.utils.tensorboard import SummaryWriter
import os
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from functools import partial

from doggelganger.models.resnet import ResNetModel
from doggelganger.train import make_training_data

def objective(config, checkpoint_dir=None):
    # Hyperparameters
    num_blocks = config["num_blocks"]
    learning_rate = config["learning_rate"]
    lambda_delta = config["lambda_delta"]
    lambda_ortho = config["lambda_ortho"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    init_method = config["init_method"]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=tune.get_trial_dir())

    # Create and train the model
    model = ResNetModel(num_blocks, learning_rate, lambda_delta, lambda_ortho, init_method=init_method)
    
    def log_metrics(epoch, loss, accuracies):
        writer.add_scalar('Loss/train', loss, epoch)
        writer.add_scalar('Accuracy/top1', accuracies[0], epoch)
        writer.add_scalar('Accuracy/top3', accuracies[1], epoch)
        writer.add_scalar('Accuracy/top10', accuracies[2], epoch)
        
        # Report intermediate result
        tune.report(loss=loss, top1_accuracy=accuracies[0], top3_accuracy=accuracies[1], top10_accuracy=accuracies[2])

    model.fit(X_train, y_train, num_epochs, batch_size, callback=log_metrics)

    # Predict on test set
    preds = model.predict(X_test)

    # Calculate accuracies
    all_y = np.vstack((y_train, y_test))
    cosine_similarities = 1 - pairwise_distances(preds, all_y, metric="cosine")
    correct_indices = np.arange(len(y_train), len(y_train) + len(y_test))

    top1_accuracy = np.mean(np.argmax(cosine_similarities, axis=1) == correct_indices)
    top3_accuracy = np.mean([np.isin(correct_indices[i], indices[:3]).any() for i, indices in enumerate(np.argsort(-cosine_similarities, axis=1))])
    top10_accuracy = np.mean([np.isin(correct_indices[i], indices[:10]).any() for i, indices in enumerate(np.argsort(-cosine_similarities, axis=1))])

    # Log final test accuracies
    writer.add_scalar('TestAccuracy/top1', top1_accuracy, 0)
    writer.add_scalar('TestAccuracy/top3', top3_accuracy, 0)
    writer.add_scalar('TestAccuracy/top10', top10_accuracy, 0)

    # Blended score (you can adjust the weights if needed)
    blended_score = 0.5 * top1_accuracy + 0.3 * top3_accuracy + 0.2 * top10_accuracy

    writer.close()

    # Final report
    tune.report(blended_score=blended_score)

def hyperparameter_search(X, y, num_samples=10, max_num_epochs=200, gpus_per_trial=0):
    config = {
        "num_blocks": tune.randint(2, 6),
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "lambda_delta": tune.loguniform(1e-3, 1e-1),
        "lambda_ortho": tune.loguniform(1e-3, 1e-1),
        "num_epochs": tune.randint(50, 201),
        "batch_size": tune.choice([16, 32, 64]),
        "init_method": tune.choice(['default', 'he', 'fixup', 'lsuv'])
    }

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    result = tune.run(
        partial(objective, X=X, y=y),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler
    )

    best_trial = result.get_best_trial("blended_score", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["blended_score"]))

    best_trained_model = ResNetModel(
        num_blocks=best_trial.config["num_blocks"],
        learning_rate=best_trial.config["learning_rate"],
        lambda_delta=best_trial.config["lambda_delta"],
        lambda_ortho=best_trial.config["lambda_ortho"],
        init_method=best_trial.config["init_method"],
    )
    best_trained_model.fit(
        X,
        y,
        num_epochs=best_trial.config["num_epochs"],
        batch_size=best_trial.config["batch_size"],
    )

    return best_trained_model, best_trial.config

def main():
    X, y = make_training_data('data/train')
    best_model, best_params = hyperparameter_search(X, y, num_samples=50)
    print("Best hyperparameters found were: ", best_params)
