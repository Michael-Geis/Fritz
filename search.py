from sentence_transformers import util
from sklearn.base import BaseEstimator, TransformerMixin
import os
import pandas as pd


class Search(BaseEstimator, TransformerMixin):
    def __init__(self, path_to_library) -> None:
        super().__init__()

        self.path_to_library = path_to_library

    def fit(self):
        return self

    def transform(self, X, y=None):
        library_metadata = pd.read_feather(
            os.path.join(self.path_to_library, "metadata.feather")
        )
        library_embeddings = pd.read_feather(
            os.path.join(self.path_to_library, "embeddings.feather")
        ).values

        matches = util.semantic_search(
            query_embeddings=X, corpus_embeddings=library_embeddings, top_k=5
        )

        recommended_indices = [dict["corpus_id"] for dict in matches[0]]

        return library_metadata.iloc[recommended_indices]
