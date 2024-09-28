import torch
import torch.nn as nn
from doggelganger.models.base import BaseModel

class MinimalPerturbationNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(MinimalPerturbationNetwork, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        
        # Initialize weights close to identity and biases to zero
        nn.init.eye_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.eye_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.relu(self.fc1(x))
        return x + self.fc2(residual)  # Residual connection

class ResNetModel(BaseModel):
    def __init__(self):
        self.model = MinimalPerturbationNetwork(384)  # Assuming DinoV2 embedding size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        
        num_epochs = 10
        batch_size = 32
        lambda_delta = 0.1
        lambda_ortho = 0.1

        for epoch in range(num_epochs):
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                loss_main = self.criterion(outputs, batch_y)
                loss_delta = torch.norm(outputs - batch_X, p=2)
                loss_ortho = torch.norm(
                    torch.mm(self.model.fc1.weight, self.model.fc1.weight.t()) - torch.eye(384).to(self.device)
                ) + torch.norm(
                    torch.mm(self.model.fc2.weight, self.model.fc2.weight.t()) - torch.eye(384).to(self.device)
                )
                
                loss = loss_main + lambda_delta * loss_delta + lambda_ortho * loss_ortho
                
                loss.backward()
                self.optimizer.step()

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
