from cleaning import cleaning
from sentence_transformers import SentenceTransformer
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import glob
import os

def embed_metadata(path_to_raw_metadata_df,path_to_save_embeddings):

    raw_metadata_df = pd.read_parquet(path_to_raw_metadata_df)
    
    clean_title = raw_metadata_df.title.apply(cleaning.cleanse)
    clean_abstract = raw_metadata_df.summary.apply(cleaning.cleanse)
    
    sentences = (clean_title + ' ' + clean_abstract).to_list()
    mini_model = SentenceTransformer('all-MiniLM-L6-v2')
    mini_sentence_embeddings = mini_model.encode(sentences=sentences,show_progress_bar=True)

    mini_sentence_embeddings_df = pd.DataFrame(mini_sentence_embeddings)
    mini_sentence_embeddings_df['sentences'] = sentences
    mini_sentence_embeddings_df.columns = mini_sentence_embeddings_df.columns.astype(str)
    table_emb = pa.table(mini_sentence_embeddings_df)
    
    pq.write_table(table_emb,path_to_save_embeddings)
    
# def embed_msc_tags(path_to_msc_dict):