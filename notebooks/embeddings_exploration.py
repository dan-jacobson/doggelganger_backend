# %%
import polars as pl
import os
# %%

uri = os.getenv("SUPABASE_DB_TEST")
query = "SELECT * FROM vecs.dog_embeddings"

embeddings = pl.read_database_uri(query=query, uri=uri)
# %%
import connectorx as cx
cx.read_sql(uri, query)                                        # read data from Postgres