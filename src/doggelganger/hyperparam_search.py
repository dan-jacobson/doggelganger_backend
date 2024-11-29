import argparse
import logging

from ray import tune
from ray.train import RunConfig, report
from ray.tune.search.optuna import OptunaSearch
from sklearn.model_selection import train_test_split

from doggelganger.models.resnet import ResNetModel
from doggelganger.train import calculate_accuracies, make_training_data

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# gotta play around with these weights
def blended_score(top1_accuracy, top3_accuracy, top10_accuracy):
    return 0.5 * top1_accuracy + 0.3 * top3_accuracy + 0.2 * top10_accuracy


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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1234)
    embedding_dim = X.shape[1]

    # Create and train the model
    model = ResNetModel(
        embedding_dim,
        num_blocks,
        learning_rate,
        lambda_delta,
        lambda_ortho,
        init_method=init_method,
    )

    def log_metrics(loss, y, preds, prefix="train"):
        top1_accuracy, top3_accuracy, top10_accuracy = calculate_accuracies(y, preds)
        score = blended_score(top1_accuracy, top3_accuracy, top10_accuracy)
        report(
            {
                f"{prefix}_loss": loss,
                f"{prefix}_top1_accuracy": top1_accuracy,
                f"{prefix}_top3_accuracy": top3_accuracy,
                f"{prefix}_top10_accuracy": top10_accuracy,
                f"{prefix}_blended_score": score,
            }
        )

    model.fit(X_train, y_train, num_epochs, batch_size, callback=None)

    # Predict on test set
    preds = model.predict(X_test)

    # Calculate accuracies
    top1_accuracy, top3_accuracy, top10_accuracy = calculate_accuracies(y_test, preds)
    score = blended_score(top1_accuracy, top3_accuracy, top10_accuracy)
    return {
        "top1_accuracy": top1_accuracy,
        "top3_accuracy": top3_accuracy,
        "top10_accuracy": top10_accuracy,
        "blended_score": score,
    }


def hyperparameter_search(X, y, num_samples=10, max_num_epochs=200, name=None):
    config = {
        "num_blocks": tune.randint(2, 16),
        "learning_rate": tune.loguniform(1e-8, 1e-2),
        "lambda_delta": tune.loguniform(1e-5, 1e-1),
        "lambda_ortho": tune.loguniform(1e-5, 1e-1),
        "num_epochs": tune.randint(50, max_num_epochs + 1),
        "batch_size": tune.choice(
            [
                16,
            ]
        ),  # 32, 64]),
        "init_method": tune.choice(["default", "he", "fixup", "lsuv"]),
    }

    # scheduler = ASHAScheduler(max_t=max_num_epochs, grace_period=2, reduction_factor=2)
    algo = OptunaSearch()

    tuner = tune.Tuner(
        tune.with_parameters(train_model, X=X, y=y),
        tune_config=tune.TuneConfig(
            metric="blended_score",
            mode="max",
            # scheduler=scheduler,
            num_samples=num_samples,
            search_alg=algo,
        ),
        param_space=config,
        run_config=RunConfig(storage_path="/Users/drj/code/doggelganger_backend/ray_results", name=name),
    )
    result = tuner.fit()

    best_trial = result.get_best_result(scope="last")
    logger.info(
        f"Best trial config: {best_trial.config}"
        f"Best trial final validation accuracy: {best_trial.metrics['blended_score']}"
    )

    return best_trial.config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hyperparameter search for ResNet model.")
    parser.add_argument("--name", type=str, default=None, help="Name for the Ray experiment")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of samples for hyperparameter search",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Should the best model be saved",
    )
    args = parser.parse_args()

    X, y = make_training_data("data/train")
    best_params = hyperparameter_search(X, y, num_samples=args.num_samples, name=args.name)

    if args.save:
        best_model = ResNetModel(
            embedding_dim=X.shape[1],
            num_blocks=best_params["num_blocks"],
            learning_rate=best_params["learning_rate"],
            lambda_delta=best_params["lambda_delta"],
            lambda_ortho=best_params["lambda_ortho"],
            init_method=best_params["init_method"],
        )
        best_model.fit(
            X,
            y,
            num_epochs=best_params["num_epochs"],
            batch_size=best_params["batch_size"],
        )
        weights_path = f"weights/{args.name}.pt"
        best_model.save(weights_path)
        logger.info(f"Best model saved to: {weights_path}")
