import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from ray import tune
from ray.train import RunConfig, report
from ray.tune.schedulers import ASHAScheduler

from doggelganger.train import make_training_data
from doggelganger.models.resnet import ResNetModel


def train_model(config, X, y):
    # Hyperparameters
    num_blocks = config["num_blocks"]
    learning_rate = config["learning_rate"]
    lambda_delta = config["lambda_delta"]
    lambda_ortho = config["lambda_ortho"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    init_method = config["init_method"]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    # Create and train the model
    model = ResNetModel(
        num_blocks, learning_rate, lambda_delta, lambda_ortho, init_method=init_method
    )

    def log_metrics(epoch, loss, accuracies):
        report(
            {
                "loss": loss,
                "top1_accuracy": accuracies[0],
                "top3_accuracy": accuracies[1],
                "top10_accuracy": accuracies[2],
            }
        )

    model.fit(X_train, y_train, num_epochs, batch_size, callback=log_metrics)

    # Predict on test set
    preds = model.predict(X_test)

    # Calculate accuracies
    all_y = np.vstack((y_train, y_test))
    cosine_similarities = 1 - pairwise_distances(preds, all_y, metric="cosine")
    correct_indices = np.arange(len(y_train), len(y_train) + len(y_test))

    top1_accuracy = np.mean(np.argmax(cosine_similarities, axis=1) == correct_indices)
    top3_accuracy = np.mean(
        [
            np.isin(correct_indices[i], indices[:3]).any()
            for i, indices in enumerate(np.argsort(-cosine_similarities, axis=1))
        ]
    )
    top10_accuracy = np.mean(
        [
            np.isin(correct_indices[i], indices[:10]).any()
            for i, indices in enumerate(np.argsort(-cosine_similarities, axis=1))
        ]
    )

    # Blended score (gotta play with these weights)
    blended_score = 0.5 * top1_accuracy + 0.3 * top3_accuracy + 0.2 * top10_accuracy

    return {
        "blended_score": blended_score,
        "top1_accuracy": top1_accuracy,
        "top3_accuracy": top3_accuracy,
        "top10_accuracy": top10_accuracy,
    }


def hyperparameter_search(X, y, num_samples=10, max_num_epochs=200):
    config = {
        "num_blocks": tune.randint(2, 6),
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "lambda_delta": tune.loguniform(1e-3, 1e-1),
        "lambda_ortho": tune.loguniform(1e-3, 1e-1),
        "num_epochs": tune.randint(50, max_num_epochs + 1),
        "batch_size": tune.choice([16, 32, 64]),
        "init_method": tune.choice(["default", "he", "fixup", "lsuv"]),
    }

    scheduler = ASHAScheduler(max_t=max_num_epochs, grace_period=2, reduction_factor=2)

    tuner = tune.Tuner(
        tune.with_parameters(train_model, X=X, y=y),
        tune_config=tune.TuneConfig(
            metric="blended_score",
            mode="max",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
        run_config=RunConfig(name="weight_init"),
    )
    result = tuner.fit()

    best_trial = result.get_best_result()
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result["loss"]}")
    print(
        f"Best trial final validation accuracy: {best_trial.last_result["blended_score"]}"
    )

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


if __name__ == "__main__":
    X, y = make_training_data("data/train")
    best_model, best_params = hyperparameter_search(X, y, num_samples=50)
    print("Best hyperparameters found were: ", best_params)
