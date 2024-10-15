# %% [markdown]
# # Model Comparison Notebook
# 
# This notebook compares the performance of different models (Linear, XGBoost, and ResNet) in predicting the top 3 dogs for given human images.

# %%
import sys
import os
sys.path.append(os.path.abspath('../'))

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from src.doggelganger.utils import get_embedding, load_model as load_embedding_model
from src.doggelganger.models import LinearRegressionModel, XGBoostModel, ResNetModel
from src.doggelganger.train import make_training_data

# %%
# Load the embedding model
embedding_model = load_embedding_model()

# Load the training data
X, y = make_training_data("../data/train")

# Load the test data (assuming you have a separate test set)
# X_test, y_test = make_training_data("../data/test")

# Load the trained models
linear_model = LinearRegressionModel.load("../weights/linear.json")
xgb_model = XGBoostModel.load("../weights/xgb.json")
resnet_model = ResNetModel.load("../weights/prodv0.2.pt", embedding_dim=X.shape[1])
resnet_model.model.to('cpu')

models = {
    "Linear": linear_model,
    "XGBoost": xgb_model,
    "ResNet": resnet_model
}

# %%
def get_top_k_dogs(human_embedding, animal_embeddings, k=3):
    similarities = cosine_similarity(human_embedding.reshape(1, -1), animal_embeddings)[0]
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    return top_k_indices

def plot_images(images, titles, main_title):
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    fig.suptitle(main_title, fontsize=16)

    if len(images) == 1:
        axes = [axes]
    
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_top_3_dogs(human_image_path, models, animal_embeddings):
    human_embedding = get_embedding(human_image_path, embedding_model)
    human_embedding = np.array(human_embedding)
    
    human_image = Image.open(human_image_path)
    
    plot_images([human_image], ["Human"], "Input Human Image")
    
    for model_name, model in models.items():
        predicted_embedding = model.predict(human_embedding.reshape(1, -1))[0]
        top_3_indices = get_top_k_dogs(predicted_embedding, animal_embeddings)
        
        top_3_images = []
        top_3_titles = []
        
        for i, idx in enumerate(top_3_indices, 1):
            animal_image_path = f"../data/train/animal/{idx:04d}"
            animal_image = None
            for ext in ['.png', '.jpg', '.jpeg']:
                try:
                    animal_image = Image.open(f"{animal_image_path}{ext}")
                    break
                except FileNotFoundError:
                    continue
            
            if animal_image:
                top_3_images.append(animal_image)
                top_3_titles.append(f"Top {i}")
            else:
                print(f"Warning: Could not find image for index {idx}")
        
        plot_images(top_3_images, top_3_titles, f"{model_name} Model - Top 3 Dogs")

# %%
# Visualize top 3 dogs for a few sample human images
import glob

sample_human_images = []
for i in range(1, 4):  # Assuming you want to keep 3 sample images
    pattern = f"../data/test/human/{i:04d}.*"
    matching_files = glob.glob(pattern)
    if matching_files:
        sample_human_images.append(matching_files[0])

for human_image_path in sample_human_images:
    visualize_top_3_dogs(human_image_path, models, y)

# %% [markdown]
# ## Model Performance Comparison
# 
# Let's compare the performance of the different models using various metrics.

# %%
from sklearn.metrics import mean_squared_error, r2_score
from src.doggelganger.train import calculate_accuracies

def evaluate_model(model, X, y):
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    top1_acc, top3_acc, top10_acc = calculate_accuracies(y, predictions)
    
    return {
        "MSE": mse,
        "R2": r2,
        "Top-1 Accuracy": top1_acc,
        "Top-3 Accuracy": top3_acc,
        "Top-10 Accuracy": top10_acc
    }

# Use the training data for evaluation (X and y)
results = {}
for model_name, model in models.items():
    results[model_name] = evaluate_model(model, X, y)

# Display results
for model_name, metrics in results.items():
    print(f"\n{model_name} Model:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

# %%
# Visualize model performance comparison
metrics = list(results[list(results.keys())[0]].keys())
x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))

for i, (model_name, model_results) in enumerate(results.items()):
    values = list(model_results.values())
    ax.bar(x + i * width, values, width, label=model_name)

ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.show()


