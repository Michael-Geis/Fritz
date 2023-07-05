import data_cleaning as clean
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import json


class embed:
    """A class to handle creating sentence transformer embeddings of arxiv titles and abstracts."""

    def prepare_sentences(dataset=pd.DataFrame()):
        """cleans title and abstract of each paper and concatenates them.

        Args:
            dataset: arxiv dataset

        Returns:
            list in which entry i is cleaned and concatenated title and abstract of article i.
        """

        clean_dataset = clean.clean_title_abstracts(dataset)
        return (clean_dataset.title + " " + clean_dataset.abstract).to_list()

    def create_sentence_embeddings(self, dataset, model_name):
        model = SentenceTransformer(model_name)
        sentences = self.prepare_sentences(dataset)
        embedding_array = model.encode(sentences=sentences, show_progress_bar=True)

        return pd.DataFrame(embedding_array).join(dataset.id)

    ## Create series object in which each entry is NAN or the list of embedded tags

    def rank_msc_tags(self, dataset):
        tag_map = clean.msc_encoded_dict()
        # Get the list of embedded tags for all tagged rows in a new column
        embedded_tags = dataset.msc_tags
        dataset['embedded_tags'] = embedded_tags[
            dataset.msc_tags.notna()
        ].apply(lambda x: [tag_map[tag] for tag in x])

        ## Finish this tomorrow

        dataset['semantic_tag_score'] = dataset.apply( ,axis=1) 
