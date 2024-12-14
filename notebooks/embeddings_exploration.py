# %%
import sqlalchemy as sa
import ast
import numpy as np
import polars as pl

uri = os.getenv("SUPABASE_DB_TEST")
query = "SELECT * FROM vecs.dog_embeddings"

# Create the engine
engine = sa.create_engine(uri)

# Execute the query
with engine.connect() as connection:
    # Execute the query and fetch all results
    result = connection.execute(sa.text(query))
    
    # Convert to a list of dictionaries
    data = [
        {
            'id': row.id, 
            'vec': ast.literal_eval(row.vec),
            'age': row.metadata.get('age'),
            'sex': row.metadata.get('sex'),
            'name': row.metadata.get('name'),
            'breed': row.metadata.get('breed')
        } 
        for row in result
    ]

# Create Polars DataFrame
embeddings = pl.DataFrame(data)
# %%
embeddings[:5]
# %%
import umap
import polars as pl
import numpy as np
import plotly.express as px
import plotly.graph_objs as go

# Assuming your dataframe is called 'embeddings'
# Convert the vector column to a numpy array
X = np.array(embeddings['vec'].to_list())

# Perform UMAP reduction
reducer = umap.UMAP(n_components=3, random_state=42)
umap_embedding = reducer.fit_transform(X)

# Create a new dataframe with UMAP coordinates
umap_df = pl.DataFrame({
    'x': umap_embedding[:, 0],
    'y': umap_embedding[:, 1],
    'z': umap_embedding[:, 2],
    'id': embeddings['id'],
    'breed': embeddings['breed'],
    'age': embeddings['age'],
    'sex': embeddings['sex'],
    'name': embeddings['name']
})

# Interactive 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=umap_df['x'],
    y=umap_df['y'],
    z=umap_df['z'],
    mode='markers',
    marker=dict(
        size=5,
        color=umap_df['breed'].cast(pl.Categorical).to_physical(),  # Convert breeds to numeric categories
        colorscale='Viridis',
        opacity=0.8
    ),
    text=[f"ID: {id}<br>Breed: {breed}<br>Name: {name}<br>Age: {age}<br>Sex: {sex}" 
          for id, breed, name, age, sex in zip(umap_df['id'], umap_df['breed'], umap_df['name'], umap_df['age'], umap_df['sex'])],
    hoverinfo='text'
)])

fig.update_layout(title='UMAP Embedding Visualization')
fig.show()
# %%
