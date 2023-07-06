import regex
import pandas as pd
import json
import sentence_transformers.util
import numpy as np
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin


class TextCleaner(BaseEstimator, TransformerMixin):
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


def arxiv_subject_dict():
    """Maps arXiv subject categories to their full english names.

    Returns:
        Python dict whose keys are arXiv tags and whose values are their English names.
        Note that the list is not exhaustive in the sense that many categories have aliases that
        are not included. (Some are, e.g. math.MP and math-ph).
    """
    return {
        "astro-ph": "Astrophysics",
        "astro-ph.CO": "Cosmology and Nongalactic Astrophysics",
        "astro-ph.EP": "Earth and Planetary Astrophysics",
        "astro-ph.GA": "Astrophysics of Galaxies",
        "astro-ph.HE": "High Energy Astrophysical Phenomena",
        "astro-ph.IM": "Instrumentation and Methods for Astrophysics",
        "astro-ph.SR": "Solar and Stellar Astrophysics",
        "cond-mat.dis-nn": "Disordered Systems and Neural Networks",
        "cond-mat.mes-hall": "Mesoscale and Nanoscale Physics",
        "cond-mat.mtrl-sci": "Materials Science",
        "cond-mat.other": "Other Condensed Matter",
        "cond-mat.quant-gas": "Quantum Gases",
        "cond-mat.soft": "Soft Condensed Matter",
        "cond-mat.stat-mech": "Statistical Mechanics",
        "cond-mat.str-el": "Strongly Correlated Electrons",
        "cond-mat.supr-con": "Superconductivity",
        "cond-mat": "Condensed Matter",
        "cs.AI": "Artificial Intelligence",
        "cs.AR": "Hardware Architecture",
        "cs.CC": "Computational Complexity",
        "cs.CE": "Computational Engineering, Finance, and Science",
        "cs.CG": "Computational Geometry",
        "cs.CL": "Computation and Language",
        "cs.CR": "Cryptography and Security",
        "cs.CV": "Computer Vision and Pattern Recognition",
        "cs.CY": "Computers and Society",
        "cs.DB": "Databases",
        "cs.DC": "Distributed, Parallel, and Cluster Computing",
        "cs.DL": "Digital Libraries",
        "cs.DM": "Discrete Mathematics",
        "cs.DS": "Data Structures and Algorithms",
        "cs.ET": "Emerging Technologies",
        "cs.FL": "Formal Languages and Automata Theory",
        "cs.GL": "General Literature",
        "cs.GR": "Graphics",
        "cs.GT": "Computer Science and Game Theory",
        "cs.HC": "Human-Computer Interaction",
        "cs.IR": "Information Retrieval",
        "cs.IT": "Information Theory",
        "cs.LG": "Machine Learning",
        "cs.LO": "Logic in Computer Science",
        "cs.MA": "Multiagent Systems",
        "cs.MM": "Multimedia",
        "cs.MS": "Mathematical Software",
        "cs.NA": "Numerical Analysis",
        "cs.NE": "Neural and Evolutionary Computing",
        "cs.NI": "Networking and Internet Architecture",
        "cs.OH": "Other Computer Science",
        "cs.OS": "Operating Systems",
        "cs.PF": "Performance",
        "cs.PL": "Programming Languages",
        "cs.RO": "Robotics",
        "cs.SC": "Symbolic Computation",
        "cs.SD": "Sound",
        "cs.SE": "Software Engineering",
        "cs.SI": "Social and Information Networks",
        "cs.SY": "Systems and Control",
        "econ.EM": "Econometrics",
        "econ.GN": "General Economics",
        "econ.TH": "Theoretical Economics",
        "eess.AS": "Audio and Speech Processing",
        "eess.IV": "Image and Video Processing",
        "eess.SP": "Signal Processing",
        "eess.SY": "Systems and Control",
        "dg-ga": "Differential Geometry",
        "gr-qc": "General Relativity and Quantum Cosmology",
        "hep-ex": "High Energy Physics - Experiment",
        "hep-lat": "High Energy Physics - Lattice",
        "hep-ph": "High Energy Physics - Phenomenology",
        "hep-th": "High Energy Physics - Theory",
        "math.AC": "Commutative Algebra",
        "math.AG": "Algebraic Geometry",
        "math.AP": "Analysis of PDEs",
        "math.AT": "Algebraic Topology",
        "math.CA": "Classical Analysis and ODEs",
        "math.CO": "Combinatorics",
        "math.CT": "Category Theory",
        "math.CV": "Complex Variables",
        "math.DG": "Differential Geometry",
        "math.DS": "Dynamical Systems",
        "math.FA": "Functional Analysis",
        "math.GM": "General Mathematics",
        "math.GN": "General Topology",
        "math.GR": "Group Theory",
        "math.GT": "Geometric Topology",
        "math.HO": "History and Overview",
        "math.IT": "Information Theory",
        "math.KT": "K-Theory and Homology",
        "math.LO": "Logic",
        "math.MG": "Metric Geometry",
        "math.MP": "Mathematical Physics",
        "math.NA": "Numerical Analysis",
        "math.NT": "Number Theory",
        "math.OA": "Operator Algebras",
        "math.OC": "Optimization and Control",
        "math.PR": "Probability",
        "math.QA": "Quantum Algebra",
        "math.RA": "Rings and Algebras",
        "math.RT": "Representation Theory",
        "math.SG": "Symplectic Geometry",
        "math.SP": "Spectral Theory",
        "math.ST": "Statistics Theory",
        "math-ph": "Mathematical Physics",
        "funct-an": "Functional Analysis",
        "alg-geom": "Algebraic Geometry",
        "nlin.AO": "Adaptation and Self-Organizing Systems",
        "chao-dyn": "Chaotic Dynamics",
        "nlin.CD": "Chaotic Dynamics",
        "nlin.CG": "Cellular Automata and Lattice Gases",
        "nlin.PS": "Pattern Formation and Solitons",
        "nlin.SI": "Exactly Solvable and Integrable Systems",
        "nucl-ex": "Nuclear Experiment",
        "nucl-th": "Nuclear Theory",
        "physics.acc-ph": "Accelerator Physics",
        "physics.ao-ph": "Atmospheric and Oceanic Physics",
        "physics.app-ph": "Applied Physics",
        "physics.atm-clus": "Atomic and Molecular Clusters",
        "physics.atom-ph": "Atomic Physics",
        "physics.bio-ph": "Biological Physics",
        "physics.chem-ph": "Chemical Physics",
        "physics.class-ph": "Classical Physics",
        "physics.comp-ph": "Computational Physics",
        "physics.data-an": "Data Analysis, Statistics and Probability",
        "physics.ed-ph": "Physics Education",
        "physics.flu-dyn": "Fluid Dynamics",
        "physics.gen-ph": "General Physics",
        "physics.geo-ph": "Geophysics",
        "physics.hist-ph": "History and Philosophy of Physics",
        "physics.ins-det": "Instrumentation and Detectors",
        "physics.med-ph": "Medical Physics",
        "physics.optics": "Optics",
        "physics.plasm-ph": "Plasma Physics",
        "physics.pop-ph": "Popular Physics",
        "physics.soc-ph": "Physics and Society",
        "physics.space-ph": "Space Physics",
        "q-bio.BM": "Biomolecules",
        "q-bio.CB": "Cell Behavior",
        "q-bio.GN": "Genomics",
        "q-bio.MN": "Molecular Networks",
        "q-bio.NC": "Neurons and Cognition",
        "q-bio.OT": "Other Quantitative Biology",
        "q-bio.PE": "Populations and Evolution",
        "q-bio.QM": "Quantitative Methods",
        "q-bio.SC": "Subcellular Processes",
        "q-bio.TO": "Tissues and Organs",
        "q-fin.CP": "Computational Finance",
        "q-fin.EC": "Economics",
        "q-fin.GN": "General Finance",
        "q-fin.MF": "Mathematical Finance",
        "q-fin.PM": "Portfolio Management",
        "q-fin.PR": "Pricing of Securities",
        "q-fin.RM": "Risk Management",
        "q-fin.ST": "Statistical Finance",
        "q-fin.TR": "Trading and Market Microstructure",
        "quant-ph": "Quantum Physics",
        "q-alg": "Quantum Algebra",
        "stat.AP": "Applications",
        "stat.CO": "Computation",
        "stat.ME": "Methodology",
        "stat.ML": "Machine Learning",
        "stat.OT": "Other Statistics",
        "stat.TH": "Statistics Theory",
    }


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
