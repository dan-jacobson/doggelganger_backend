import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModel

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

class EmbeddingDataset(Dataset):
    def __init__(self, selfie_embeddings, dog_embeddings):
        self.selfie_embeddings = selfie_embeddings
        self.dog_embeddings = dog_embeddings

    def __len__(self):
        return len(self.selfie_embeddings)

    def __getitem__(self, idx):
        return self.selfie_embeddings[idx], self.dog_embeddings[idx]

def generate_embeddings(image_paths, model, feature_extractor, device):
    embeddings = []
    for path in image_paths:
        image = Image.open(path).convert('RGB')
        inputs = feature_extractor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0].cpu()  # Use CLS token embedding
        embeddings.append(embedding)
    return torch.cat(embeddings, dim=0)

def train_minimal_perturbation_network(selfie_paths, dog_paths, num_epochs=10, batch_size=32, lambda_delta=0.1, lambda_ortho=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load DinoV2 model
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/dinov2-base")
    dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)

    # Generate embeddings
    selfie_embeddings = generate_embeddings(selfie_paths, dino_model, feature_extractor, device)
    dog_embeddings = generate_embeddings(dog_paths, dino_model, feature_extractor, device)

    # Create dataset and dataloader
    dataset = EmbeddingDataset(selfie_embeddings, dog_embeddings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the transfer network
    embedding_dim = selfie_embeddings.shape[1]
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

    return transfer_net, dino_model, feature_extractor