import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

from doggelganger.utils import get_embedding, load_model as load_embedding_model
from doggelganger.train import make_training_data
from doggelganger.models.resnet import MinimalPerturbationNetwork

EPOCHS = 100
BATCH_SIZE = 32
LAMBDA_DELTA = 0.1
LAMBDA_ORTHO = 0.1
TEST_SIZE = 0.1

class EmbeddingDataset(Dataset):
    def __init__(self, selfie_embeddings, dog_embeddings):
        self.selfie_embeddings = selfie_embeddings
        self.dog_embeddings = dog_embeddings

    def __len__(self):
        return len(self.selfie_embeddings)

    def __getitem__(self, idx):
        return self.selfie_embeddings[idx], self.dog_embeddings[idx]

def main(num_epochs=EPOCHS, batch_size=BATCH_SIZE, lambda_delta=LAMBDA_DELTA, lambda_ortho=LAMBDA_ORTHO, test_size=TEST_SIZE):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Set up TensorBoard
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs', current_time)
    writer = SummaryWriter(log_dir)

    X, y = make_training_data("/data/train")

    # Create dataset
    dataset = EmbeddingDataset(X, y)

    # Split dataset into train and test sets
    test_size = int(len(dataset) * test_size)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the transfer network
    embedding_dim = 384
    transfer_net = MinimalPerturbationNetwork(embedding_dim).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(transfer_net.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        transfer_net.train()
        running_loss = 0.0
        for i, (selfie_emb, dog_emb) in enumerate(train_dataloader):
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

        avg_train_loss = running_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        # Validation
        transfer_net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for selfie_emb, dog_emb in test_dataloader:
                selfie_emb, dog_emb = selfie_emb.to(device), dog_emb.to(device)
                outputs = transfer_net(selfie_emb)
                val_loss += criterion(outputs, dog_emb).item()

        avg_val_loss = val_loss / len(test_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)

    writer.close()
    return transfer_net

if __name__=="__main__":
    main()
