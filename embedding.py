import cleaning as clean
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import json
from sklearn.base import BaseEstimator, TransformerMixin
import os


class Embedder(BaseEstimator, TransformerMixin):
    """A class to handle creating sentence transformer embeddings from a clean arxiv dataset."""

    def fit(self, X, y=None):
        return self

    def transform(
        self,
        X=None,
        y=None,
        model_name=None,
        load_from_file=False,
        path_to_embeddings=None,
    ):
        """Either generates embeddings from an clean ArXivData instance or loads embeddings from file.

        Args:
            X: ArXivData instance that has been cleaned
            y: Labels. Defaults to None.
            model_name: Sentence transformer model used to generate embeddings. Defaults to None.
            load_from_file: Boolean used to specify whether to calculate embeddings or load from file. Defaults to False.
            path_to_embeddings: path to the location to save embeddings to or load embeddings from. Defaults to None.

        Raises:
            Exception: Raises exception if the load_from_file is True without a specified path to load from.
        """

        if load_from_file:
            if not path_to_embeddings:
                raise Exception("You must specify a path to store the embeddings.")
            X.embeddings = pd.read_feather(path_to_embeddings).to_numpy()
        else:
            ## Generate embeddings from X and save as an attribute of X.

            if not model_name:
                raise Exception(
                    "You must specify the sentence transformer model to use."
                )

            doc_strings = (X.metadata.doc_strings).to_list()
            model = SentenceTransformer(model_name)
            embeddings = model.encode(doc_strings, show_progress_bar=True)
            X.embeddings = embeddings

            ## Save the embeddings to the specified path, or, if no path is specified, use the default path
            ## default path = ./model_name_embeddings.feather

            embeddings_df = pd.DataFrame(embeddings)
            embeddings_df.columns = [
                str(column_name) for column_name in embeddings_df.columns
            ]

            if not path_to_embeddings:
                path_to_embeddings = os.path.join(
                    os.getcwd(), f"{model_name}_embeddings.feather"
                )

            embeddings_df.to_feather(path_to_embeddings)


class ComputeMSCLabels(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, path_to_embeddings=None):
        tag_to_embedding_dict = clean.msc_encoded_dict()

        X["scored_tags"] = np.nan

        X_tagged_rows = X[X.msc_tags.notna()]

        X_tagged_rows["tag_embeddings"] = X_tagged_rows.msc_tags.apply(
            clean.list_mapper, dictionary=tag_to_embedding_dict
        )
        tag_scores = X_tagged_rows.apply(
            self.get_tag_semantic_scores, path_to_embeddings=path_to_embeddings, axis=1
        )
        X.scored_tags[X.metadata.msc_tags.notna()] = tag_scores

        return X

    def get_tag_semantic_scores(self, metadata_row, path_to_embeddings):
        embeddings = pd.read_feather(path_to_embeddings).to_numpy()
        results = util.semantic_search(
            query_embeddings=list(embeddings[metadata_row.doc_strings.index, :]),
            corpus_embeddings=metadata_row.tag_embeddings,
            top_k=50,
        )

        return results[0]


def generate_tag_embeddings(model_name, path_to_tag_dict, path_to_save_embeddings):
    model = SentenceTransformer(model_name)
    with open(path_to_tag_dict, "r") as file:
        dict_string = file.read()
        tag_dict = json.loads(dict_string)

    tag_name_list = list(tag_dict.values())
    embedded_tag_names = model.encode(sentences=tag_name_list, show_progress_bar=True)
    embedded_tag_names_df = pd.DataFrame(embedded_tag_names)
    embedded_tag_names_df.columns = [
        str(name) for name in embedded_tag_names_df.columns
    ]

    embedded_tag_names_df.to_feather(path_to_save_embeddings)


def load_tag_embeddings(path_to_tag_dict, path_to_tag_embeddings):
    with open(path_to_tag_dict, "r") as file:
        dict_string = file.read()
        tag_dict = json.loads(dict_string)

    tag_name_list = list(tag_dict.values())
    tag_name_embeddings = pd.read_feather(path_to_tag_embeddings)
    return tag_name_embeddings.reindex(index=tag_name_list)
