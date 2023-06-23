import regex
import pandas as pd
import json
import sentence_transformers.util
import os

def main(raw_metadata_df, path_to_embeddings):
    clean_metadata_df = pd.DataFrame(
        columns=['sentences','authors','msc_tags','msc_cos_sim']
        )

    clean_title = raw_metadata_df.title.apply(cleanse)
    clean_abstract = raw_metadata_df.summary.apply(cleanse)
    clean_metadata_df.sentences = clean_title + ' ' + clean_abstract
    clean_metadata_df.authors = raw_metadata_df.authors
    clean_metadata_df.msc_tags = raw_metadata_df.categories.apply(cats_to_msc)

    return clean_metadata_df

##


## 1. Latin-ize latex accents enclosed in brackets
def remove_latex_accents(string):
    accent = r'\\[\'\"\^\`H\~ckl=bdruvtoi]\{([a-z])\}'
    replacement = r'\1'

    string = regex.sub(accent,replacement, string)
    return string

## 2. Remove latex environments
def remove_env(string):
    env = r'\\[a-z]{2,}{[^{}]+?}'

    string = regex.sub(env,'',string)
    return string

## 3. Latin-ize non-{} enclosed latex accents:
def remove_accents(string):
    accent = r'\\[\'\"\^\`H\~ckl=bdruvtoi]([a-z])'
    replacement = r'\1'

    string = regex.sub(accent,replacement,string)
    return string 

## 4. ONLY remove latex'd math that is separated as a 'word' i.e. has space characters on either side of it.

def remove_latex(string):
    latex = r'\s(\$\$?)[^\$]*?\1\S*'
    string = regex.sub(latex,' LATEX ',string)
    return string 


def cleanse(string):
    string = string.replace('\n',' ')
    string = remove_latex_accents(string)
    string = remove_env(string)
    string = remove_accents(string)
    string = remove_latex(string)
    return string

## 

def find_msc(cat_list):
    pattern = r'\b\d{2}[0-9a-zA-Z]{3}\b'
    out = []
    for cat in cat_list:
        tags = regex.findall(pattern,cat)
        for tag in tags:
            out.append(tag)
    return out

def msc_tags():
    with open('./data/msc.json','r') as file:
        text = file.read()
        return json.loads(text)


def cats_to_msc(cat_list):
    out = []
    for tag in find_msc(cat_list):
        if tag in msc_tags().keys():
            out.append(msc_tags()[tag])
        else:
            continue
    if out == []:
        return None
    else:
        return out
    

##

def msc_encoded_dict():
    encoded_tags = pd.read_parquet('./data/msc_mini_embeddings.parquet').to_numpy()
    return {k : v for (k,v) in zip(msc_tags().values(), encoded_tags)}

def doc_encoded_dict():
    library_embeddings = pd.read_parquet('./data/APSP_mini_vec.parquet')

    docs = library_embeddings.docs.to_list()
    encoded_docs = library_embeddings.vecs.to_numpy()

    return {k : v for (k,v) in zip(docs , encoded_docs)}

def score_tags(processed_arxiv_row):

    tag_list = processed_arxiv_row.msc_tags
    title_plus_abstract = processed_arxiv_row.docs

    if tag_list is None:
        return None
    embedded_msc_tags = [msc_encoded_dict()[tag] for tag in tag_list]
    
    return sentence_transformers.util.semantic_search(
        query_embeddings=doc_encoded_dict()[title_plus_abstract],
        corpus_embeddings=embedded_msc_tags,
        )[0]
    
