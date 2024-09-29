import optuna
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances

def objective(trial, model_class, X, y):
    # Hyperparameters to optimize
    num_blocks = trial.suggest_int("num_blocks", 2, 5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    lambda_delta = trial.suggest_float("lambda_delta", 1e-3, 1e-1, log=True)
    lambda_ortho = trial.suggest_float("lambda_ortho", 1e-3, 1e-1, log=True)
    num_epochs = trial.suggest_int("num_epochs", 50, 200)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = model_class(num_blocks, learning_rate, lambda_delta, lambda_ortho)
    model.fit(X_train, y_train, num_epochs, batch_size)

    # Predict on test set
    preds = model.predict(X_test)

    # Calculate accuracies
    all_y = np.vstack((y_train, y_test))
    cosine_similarities = 1 - pairwise_distances(preds, all_y, metric="cosine")
    correct_indices = np.arange(len(y_train), len(y_train) + len(y_test))

    top1_accuracy = np.mean(np.argmax(cosine_similarities, axis=1) == correct_indices)
    top3_accuracy = np.mean([np.isin(correct_indices[i], indices[:3]).any() for i, indices in enumerate(np.argsort(-cosine_similarities, axis=1))])
    top10_accuracy = np.mean([np.isin(correct_indices[i], indices[:10]).any() for i, indices in enumerate(np.argsort(-cosine_similarities, axis=1))])

    # Blended score (you can adjust the weights if needed)
    blended_score = 0.5 * top1_accuracy + 0.3 * top3_accuracy + 0.2 * top10_accuracy

    return blended_score

def hyperparameter_search(model_class, X, y, n_trials=100):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, model_class, X, y), n_trials=n_trials)

    best_params = study.best_params
    best_model = model_class(
        num_blocks=best_params["num_blocks"],
        learning_rate=best_params["learning_rate"],
        lambda_delta=best_params["lambda_delta"],
        lambda_ortho=best_params["lambda_ortho"],
    )
    best_model.fit(
        X,
        y,
        num_epochs=best_params["num_epochs"],
        batch_size=best_params["batch_size"],
    )
    return best_model, best_params
