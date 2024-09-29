import torch
import torch.nn as nn
from doggelganger.models.base import BaseModel
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import optuna


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.relu(self.fc1(x))
        return x + self.fc2(residual)


class MinimalPerturbationNetwork(nn.Module):
    def __init__(self, embedding_dim, num_blocks=3):
        super(MinimalPerturbationNetwork, self).__init__()
        self.blocks = nn.ModuleList(
            [ResidualBlock(embedding_dim) for _ in range(num_blocks)]
        )
        self.init_weights()

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.normal_(param, mean=0, std=1e-3)  # Very small initial weights
            elif "bias" in name:
                nn.init.constant_(param, 0)


class ResNetModel(BaseModel):
    def __init__(
        self, num_blocks=3, learning_rate=0.001, lambda_delta=0.1, lambda_ortho=0.1
    ):
        self.num_blocks = num_blocks
        self.model = MinimalPerturbationNetwork(
            384, num_blocks
        )  # Assuming DinoV2 embedding size
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.writer = SummaryWriter()
        self.lambda_delta = lambda_delta
        self.lambda_ortho = lambda_ortho

    def fit(self, X, y, num_epochs=100, batch_size=32):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)

        global_step = 0
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = 0
            for i in range(0, len(X), batch_size):
                batch_X = X[i : i + batch_size]
                batch_y = y[i : i + batch_size]

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)

                loss_main = self.criterion(outputs, batch_y)
                loss_delta = torch.norm(outputs - batch_X, p=2)
                loss_ortho = 0
                for block in self.model.blocks:
                    loss_ortho += torch.norm(
                        torch.mm(block.fc1.weight, block.fc1.weight.t())
                        - torch.eye(384).to(self.device)
                    ) + torch.norm(
                        torch.mm(block.fc2.weight, block.fc2.weight.t())
                        - torch.eye(384).to(self.device)
                    )

                loss = (
                    loss_main
                    + self.lambda_delta * loss_delta
                    + self.lambda_ortho * loss_ortho
                )

                loss.backward()
                self.optimizer.step()

                # Log metrics
                self.writer.add_scalar("Loss/total", loss.item(), global_step)
                self.writer.add_scalar("Loss/main", loss_main.item(), global_step)
                self.writer.add_scalar("Loss/delta", loss_delta.item(), global_step)
                self.writer.add_scalar("Loss/ortho", loss_ortho.item(), global_step)

                epoch_loss += loss.item()
                global_step += 1

            # Log epoch average loss
            self.writer.add_scalar(
                "Loss/epoch", epoch_loss / (len(X) // batch_size), epoch
            )

        self.writer.close()
        return epoch_loss / (len(X) // batch_size)

    @staticmethod
    def objective(trial, X, y):
        num_blocks = trial.suggest_int("num_blocks", 2, 5)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        lambda_delta = trial.suggest_float("lambda_delta", 1e-3, 1e-1, log=True)
        lambda_ortho = trial.suggest_float("lambda_ortho", 1e-3, 1e-1, log=True)
        num_epochs = trial.suggest_int("num_epochs", 50, 200)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        model = ResNetModel(num_blocks, learning_rate, lambda_delta, lambda_ortho)
        final_loss = model.fit(X, y, num_epochs, batch_size)
        return final_loss

    @classmethod
    def hyperparameter_search(cls, X, y, n_trials=100):
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: cls.objective(trial, X, y), n_trials=n_trials)

        best_params = study.best_params
        best_model = cls(
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

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            return self.model(X).cpu().numpy()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    @staticmethod
    def load(path):
        model = ResNetModel()
        model.model.load_state_dict(torch.load(path))
        return model
