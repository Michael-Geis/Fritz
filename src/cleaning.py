import regex
import pandas as pd
import json
import sentence_transformers.util
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin


class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        doc_strings = (
            X.title.apply(cleanse) + " " + X.abstract.apply(cleanse)
        ).to_list()

        return doc_strings


class FullTextCleaner(BaseEstimator, TransformerMixin):
    """Return ArXivData class object with its metadata attribute modified so that
    1. The 'title' and 'abstract' columns have been scrubbed of latex and accented characters
    2. The msc tag list has been translated to english.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.metadata.title = X.metadata.title.apply(cleanse)
        X.metadata.abstract = X.metadata.abstract.apply(cleanse)
        X.metadata.msc_tags[X.metadata.msc_tags.notna()] = X.metadata.msc_tags[
            X.metadata.msc_tags.notna()
        ].apply(list_mapper, dictionary=msc_tags())
        X.metadata["doc_strings"] = X.metadata.title + " " + X.metadata.abstract

        return X


def arxiv_subjects():
    with open("./data/arxiv_subjects.json", "r") as file:
        dictionary = file.read()
        return json.loads(dictionary)


def msc_tags():
    with open("./data/msc.json", "r") as file:
        dictionary = file.read()
        return json.loads(dictionary)


def list_mapper(item_list, dictionary):
    mapped_item_list = [
        dictionary[item] for item in item_list if item in dictionary.keys()
    ]
    if len(mapped_item_list) == 0:
        return None
    else:
        return mapped_item_list


def split_categories(raw_metadata):
    """Takes in raw metadata returned by an ArXiv query and converts the 'categories' column into separate
    arxiv subject tags and msc tags.

    Args:
        raw_metadata: Dataframe returned by the `data_storage.query_to_df` method. Raw ArXiv query results.

    Returns:
        The input dataframe with the 'categories' column removed and replaced by 'arxiv_subjects' which is a
        list of the arxiv subject tags in the categories list, and 'msc_tags' which is a list of the msc tags
        in the categories list.
    """
    split_metadata = raw_metadata.copy().drop(columns=["categories"])
    split_metadata["arxiv_subjects"] = extract_arxiv_subjects(raw_metadata)
    split_metadata["msc_tags"] = extract_msc_tags(raw_metadata)
    return split_metadata


def OHE_arxiv_subjects(metadata):
    mlb = MultiLabelBinarizer()
    OHE_subject_array = mlb.fit_transform(metadata.arxiv_subjects)

    OHE_arxiv_subjects = pd.DataFrame(data=OHE_subject_array, columns=mlb.classes_)

    mapper = arxiv_subjects()
    OHE_arxiv_subjects = OHE_arxiv_subjects.rename(columns=mapper)
    OHE_arxiv_subjects = OHE_arxiv_subjects.loc[
        :, ~OHE_arxiv_subjects.columns.duplicated()
    ]
    return OHE_arxiv_subjects


def extract_arxiv_subjects(raw_metadata):
    def get_arxiv_subjects_from_cats(categories):
        return [tag for tag in categories if tag in arxiv_subjects().keys()]

    return raw_metadata.categories.apply(get_arxiv_subjects_from_cats)


def extract_msc_tags(raw_metadata):
    ## Check the last entry for 5 digit msc tags only.

    msc_tags = raw_metadata.categories.apply(lambda x: find_msc(x[-1]))

    msc_tags = msc_tags.apply(lambda x: np.nan if len(x) == 0 else x)

    return msc_tags


#### LATEX CLEANING UTILITIES


## 1. Latin-ize latex accents enclosed in brackets
def remove_latex_accents(string):
    accent = r"\\[\'\"\^\`H\~ckl=bdruvtoi]\{([a-z])\}"
    replacement = r"\1"

    string = regex.sub(accent, replacement, string)
    return string


## 2. Remove latex environments
def remove_env(string):
    env = r"\\[a-z]{2,}{[^{}]+?}"

    string = regex.sub(env, "", string)
    return string


## 3. Latin-ize non-{} enclosed latex accents:
def remove_accents(string):
    accent = r"\\[\'\"\^\`H\~ckl=bdruvtoi]([a-z])"
    replacement = r"\1"

    string = regex.sub(accent, replacement, string)
    return string


## 4. ONLY remove latex'd math that is separated as a 'word' i.e. has space characters on either side of it.


def remove_latex(string):
    latex = r"\s(\$\$?)[^\$]*?\1\S*"
    string = regex.sub(latex, " LATEX ", string)
    return string


def cleanse(string):
    string = string.replace("\n", " ")
    string = remove_latex_accents(string)
    string = remove_env(string)
    string = remove_accents(string)
    string = remove_latex(string)
    return string


##


def find_hyph(text):
    pattern = r"(?<!-)\b(?:\w+)(?=-)(?:-(?=\w)\w+)+(?!-)\b"
    keywords = regex.findall(pattern, text)

    if keywords == []:
        return None
    else:
        return list(set(keywords))


def find_msc(msc_string):
    five_digit_pattern = r"\b\d{2}[0-9a-zA-Z]{3}\b"
    five_digit_tags = regex.findall(five_digit_pattern, msc_string)
    return five_digit_tags


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
    encoded_tags = pd.read_parquet("./data/msc_mini_embeddings.parquet").to_numpy()
    return {k: v for (k, v) in zip(msc_tags().values(), encoded_tags)}


def doc_encoded_dict():
    library_embeddings = pd.read_parquet("./data/APSP_mini_vec.parquet")

    docs = library_embeddings.docs.to_list()
    encoded_docs = library_embeddings.vecs.to_numpy()

    return {k: v for (k, v) in zip(docs, encoded_docs)}


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
