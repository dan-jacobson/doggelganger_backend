import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from doggelganger.utils import get_embedding, load_model as load_embedding_model
from doggelganger.train import make_training_data
from doggelganger.models.resnet import MinimalPerturbationNetwork

EPOCHS = 10
BATCH_SIZE = 32
LAMBDA_DELTA = 0.1
LAMBDA_ORTHO = 0.1

class EmbeddingDataset(Dataset):
    def __init__(self, selfie_embeddings, dog_embeddings):
        self.selfie_embeddings = selfie_embeddings
        self.dog_embeddings = dog_embeddings

    def __len__(self):
        return len(self.selfie_embeddings)

    def __getitem__(self, idx):
        return self.selfie_embeddings[idx], self.dog_embeddings[idx]

def main(num_epochs=EPOCHS, batch_size=BATCH_SIZE, lambda_delta=LAMBDA_DELTA, lambda_ortho=LAMBDA_ORTHO):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    X, y = make_training_data("/data/train")

    # Create dataset and dataloader
    dataset = EmbeddingDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the transfer network
    embedding_dim = 384
    transfer_net = MinimalPerturbationNetwork(embedding_dim).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(transfer_net.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (selfie_emb, dog_emb) in enumerate(dataloader):
            selfie_emb, dog_emb = selfie_emb.to(device), dog_emb.to(device)

            optimizer.zero_grad()
            outputs = transfer_net(selfie_emb)
            
            # Main loss: match dog embeddings
            loss_main = criterion(outputs, dog_emb)
            
            # Delta norm loss: minimize changes to original embedding
            loss_delta = torch.norm(outputs - selfie_emb, p=2)
            
            # Orthogonality loss: encourage weight matrices to be orthogonal
            loss_ortho = torch.norm(
                torch.mm(transfer_net.fc1.weight, transfer_net.fc1.weight.t()) - torch.eye(embedding_dim).to(device)
            ) + torch.norm(
                torch.mm(transfer_net.fc2.weight, transfer_net.fc2.weight.t()) - torch.eye(embedding_dim).to(device)
            )
            
            # Combine losses
            loss = loss_main + lambda_delta * loss_delta + lambda_ortho * loss_ortho
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

    return transfer_net