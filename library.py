import os
import pandas as pd
from storage import query_to_df
from cleaning import TextCleaner
from sentence_transformers import SentenceTransformer


def main(library_name, query, max_results, model_name):
    path_to_library = os.path.join("./data/libraries", library_name)
    os.mkdir(path_to_library)

    ## Generate metadata from query and save
    metadata = query_to_df(query=query, max_results=max_results)
    metadata.to_feather(os.path.join(path_to_library, "metadata.feather"))

    ## Process and clean the data
    sentences = TextCleaner().transform(metadata)

    ## Generate and save embeddings
    embeddings = SentenceTransformer(model_name_or_path=model_name).encode(
        sentences=sentences, show_progress_bar=True
    )
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df.columns = [str(col_name) for col_name in embeddings_df.columns]
    embeddings_df.to_feather(os.path.join(path_to_library, "embeddings.feather"))
