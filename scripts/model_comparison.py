# %% [markdown]
# # Model Comparison Notebook
#
# This notebook compares the performance of different models (Linear, XGBoost, and ResNet) in predicting the top 3 dogs for given human images.

# %%
import os
import sys

sys.path.append(os.path.abspath("../"))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from src.doggelganger.models import LinearRegressionModel, ResNetModel, XGBoostModel
from src.doggelganger.train import make_training_data
from src.doggelganger.utils import get_embedding
from src.doggelganger.utils import load_model as load_embedding_model

# %%
# Load the embedding model
embedding_model = load_embedding_model()

# Load the training data
X, _ = make_training_data("../data/train")

# Create embeddings for dogs in carousel and example_dog_images
import glob

# def create_dog_embeddings(folders):
#     dog_embeddings = []
#     for folder in folders:
#         image_paths = glob.glob(f"{folder}/*.{{png,jpg,jpeg}}")
#         for image_path in tqdm(image_paths, desc=f"Processing {folder}"):
#             embedding = get_embedding(image_path, embedding_model)
#             dog_embeddings.append(embedding)
#     return np.array(dog_embeddings)

# y = create_dog_embeddings(["../data/carousel", "../data/example_dog_images"])

print(f"Number of dog embeddings: {len(y)}")

# Load the trained models
linear_model = LinearRegressionModel.load("../weights/linear.json")
xgb_model = XGBoostModel.load("../weights/xgb.json")
resnet_model = ResNetModel.load("../weights/prodv0.2.pt", embedding_dim=X.shape[1])
resnet_model.model.to("cpu")

models = {"Linear": linear_model, "XGBoost": xgb_model, "ResNet": resnet_model}


# %%
def get_top_k_dogs(human_embedding, animal_embeddings, k=3):
    similarities = cosine_similarity(human_embedding.reshape(1, -1), animal_embeddings)[
        0
    ]
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    return top_k_indices


def plot_images(images, titles, main_title):
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    fig.suptitle(main_title, fontsize=16)

    if len(images) == 1:
        axes = [axes]

    for i, (img, title) in enumerate(zip(images, titles, strict=False)):
        axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_top_3_dogs(
    human_image_path, models, animal_embeddings, animal_image_paths
):
    human_embedding = get_embedding(human_image_path, embedding_model)
    human_embedding = np.array(human_embedding)

    human_image = Image.open(human_image_path)

    plot_images([human_image], ["Human"], "Input Human Image")

    for model_name, model in models.items():
        predicted_embedding = model.predict(human_embedding.reshape(1, -1))[0]
        similarities = cosine_similarity(
            predicted_embedding.reshape(1, -1), animal_embeddings
        )[0]
        top_3_indices = np.argsort(similarities)[-3:][::-1]

        top_3_images = []
        top_3_titles = []

        for i, idx in enumerate(top_3_indices, 1):
            animal_image_path = animal_image_paths[idx]
            try:
                animal_image = Image.open(animal_image_path)
                top_3_images.append(animal_image)
                top_3_titles.append(f"Top {i} -- Similarity {similarities[idx]:.4f}")
            except FileNotFoundError:
                print(f"Warning: Could not find image for index {idx}")

        plot_images(top_3_images, top_3_titles, f"{model_name} Model")


def create_dog_embeddings_and_paths(folders):
    dog_embeddings = []
    dog_image_paths = []
    for folder in folders:
        image_paths = glob.glob(f"{folder}/*")
        for image_path in tqdm(image_paths, desc=f"Processing {folder}"):
            try:
                embedding = get_embedding(image_path, embedding_model)
                if embedding is not None and len(embedding) > 0:
                    dog_embeddings.append(embedding)
                    dog_image_paths.append(image_path)
                else:
                    print(f"Warning: Empty embedding for {image_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")

    if len(dog_embeddings) > 0:
        # Find the most common embedding shape
        shapes = [np.array(emb).shape for emb in dog_embeddings]
        most_common_shape = max(set(shapes), key=shapes.count)

        # Filter embeddings to keep only those with the most common shape
        filtered_embeddings = []
        filtered_paths = []
        for emb, path in zip(dog_embeddings, dog_image_paths, strict=False):
            if np.array(emb).shape == most_common_shape:
                filtered_embeddings.append(emb)
                filtered_paths.append(path)
            else:
                print(
                    f"Warning: Discarding embedding with shape {np.array(emb).shape} for {path}"
                )

        return np.array(filtered_embeddings), filtered_paths
    else:
        print("Warning: No valid embeddings found")
        return np.array([]), []


# Create embeddings and image paths for dogs in train dataset
y_train, train_image_paths = create_dog_embeddings_and_paths(["../data/train/animal"])

# Create embeddings and image paths for dogs in carousel and example_dog_images
y_carousel, carousel_image_paths = create_dog_embeddings_and_paths(
    ["../data/carousel", "../data/example_dog_images"]
)

print(f"Number of dog embeddings in train dataset: {len(y_train)}")
print(f"Number of dog embeddings in carousel and example images: {len(y_carousel)}")

# %%
# Visualize top 3 dogs for a few sample human images

sample_human_images = []
for i in range(1, 4):  # Assuming you want to keep 3 sample images
    pattern = f"../data/test/human/{i:04d}.*"
    matching_files = glob.glob(pattern)
    if matching_files:
        sample_human_images.append(matching_files[0])

print("Visualizing top 3 dogs using train dataset:")
for human_image_path in sample_human_images:
    visualize_top_3_dogs(human_image_path, models, y_train, train_image_paths)

print("\nVisualizing top 3 dogs using carousel and example_dog_images:")
for human_image_path in sample_human_images:
    visualize_top_3_dogs(human_image_path, models, y_carousel, carousel_image_paths)

# %%
import vecs
from dotenv import load_dotenv

load_dotenv()
DOGGELGANGER_DB_CONNECTION = os.getenv("SUPABASE_DB")

vx = vecs.create_client(DOGGELGANGER_DB_CONNECTION)
dogs = vx.get_or_create_collection(
    name="dog_embeddings", dimension=embedding_model.model.config.hidden_size
)

print("\nVisualizing top 3 dogs using prod dataset:")
for human_image_path in sample_human_images:
    human_embedding = get_embedding(human_image_path, embedding_model)
    human_embedding = np.array(human_embedding)

    human_image = Image.open(human_image_path)

    plot_images([human_image], ["Human"], "Input Human Image")

    for model_name, model in models.items():
        aligned_embedding = model.predict(human_embedding.reshape(1, -1))[0]
        results = dogs.query(
            data=aligned_embedding,
            limit=3,
            include_metadata=True,
            include_value=True,
        )
        # similarities = cosine_similarity(predicted_embedding.reshape(1, -1), animal_embeddings)[0]
        # top_3_indices = np.argsort(similarities)[-3:][::-1]

        top_3_images = []
        top_3_titles = []

        for i, (id, score, metadata) in enumerate(results, 1):
            animal_image_path = metadata["image_url"]
            try:
                animal_image = Image.open(animal_image_path)
                top_3_images.append(animal_image)
                top_3_titles.append(f"Top {i} -- Similarity {1 - score:.4f}")
            except FileNotFoundError:
                print(f"Warning: Could not find image for index {idx}")

        plot_images(top_3_images, top_3_titles, f"{model_name} Model")

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
        "Top-10 Accuracy": top10_acc,
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

ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison")
ax.set_xticks(x + width)
ax.set_xticklabels(metrics, rotation=45, ha="right")
ax.legend()

plt.tight_layout()
plt.show()
