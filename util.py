import os
import glob
import pandas as pd
import regex
import arxiv
import json
import util

def category_map():
    """Maps arXiv subject categories to their full english names.

    Returns:
        Python dict whose keys are arXiv tags and whose values are their English names.
        Note that the list is not exhaustive in the sense that many categories have aliases that
        are not included. (Some are, e.g. math.MP and math-ph).
    """
    return {'astro-ph': 'Astrophysics',
    'astro-ph.CO': 'Cosmology and Nongalactic Astrophysics',
    'astro-ph.EP': 'Earth and Planetary Astrophysics',
    'astro-ph.GA': 'Astrophysics of Galaxies',
    'astro-ph.HE': 'High Energy Astrophysical Phenomena',
    'astro-ph.IM': 'Instrumentation and Methods for Astrophysics',
    'astro-ph.SR': 'Solar and Stellar Astrophysics',
    'cond-mat.dis-nn': 'Disordered Systems and Neural Networks',
    'cond-mat.mes-hall': 'Mesoscale and Nanoscale Physics',
    'cond-mat.mtrl-sci': 'Materials Science',
    'cond-mat.other': 'Other Condensed Matter',
    'cond-mat.quant-gas': 'Quantum Gases',
    'cond-mat.soft': 'Soft Condensed Matter',
    'cond-mat.stat-mech': 'Statistical Mechanics',
    'cond-mat.str-el': 'Strongly Correlated Electrons',
    'cond-mat.supr-con': 'Superconductivity',
    'cond-mat': 'Condensed Matter',
    'cs.AI': 'Artificial Intelligence',
    'cs.AR': 'Hardware Architecture',
    'cs.CC': 'Computational Complexity',
    'cs.CE': 'Computational Engineering, Finance, and Science',
    'cs.CG': 'Computational Geometry',
    'cs.CL': 'Computation and Language',
    'cs.CR': 'Cryptography and Security',
    'cs.CV': 'Computer Vision and Pattern Recognition',
    'cs.CY': 'Computers and Society',
    'cs.DB': 'Databases',
    'cs.DC': 'Distributed, Parallel, and Cluster Computing',
    'cs.DL': 'Digital Libraries',
    'cs.DM': 'Discrete Mathematics',
    'cs.DS': 'Data Structures and Algorithms',
    'cs.ET': 'Emerging Technologies',
    'cs.FL': 'Formal Languages and Automata Theory',
    'cs.GL': 'General Literature',
    'cs.GR': 'Graphics',
    'cs.GT': 'Computer Science and Game Theory',
    'cs.HC': 'Human-Computer Interaction',
    'cs.IR': 'Information Retrieval',
    'cs.IT': 'Information Theory',
    'cs.LG': 'Machine Learning',
    'cs.LO': 'Logic in Computer Science',
    'cs.MA': 'Multiagent Systems',
    'cs.MM': 'Multimedia',
    'cs.MS': 'Mathematical Software',
    'cs.NA': 'Numerical Analysis',
    'cs.NE': 'Neural and Evolutionary Computing',
    'cs.NI': 'Networking and Internet Architecture',
    'cs.OH': 'Other Computer Science',
    'cs.OS': 'Operating Systems',
    'cs.PF': 'Performance',
    'cs.PL': 'Programming Languages',
    'cs.RO': 'Robotics',
    'cs.SC': 'Symbolic Computation',
    'cs.SD': 'Sound',
    'cs.SE': 'Software Engineering',
    'cs.SI': 'Social and Information Networks',
    'cs.SY': 'Systems and Control',
    'econ.EM': 'Econometrics',
    'econ.GN': 'General Economics',
    'econ.TH': 'Theoretical Economics',
    'eess.AS': 'Audio and Speech Processing',
    'eess.IV': 'Image and Video Processing',
    'eess.SP': 'Signal Processing',
    'eess.SY': 'Systems and Control',
    'dg-ga': 'Differential Geometry',
    'gr-qc': 'General Relativity and Quantum Cosmology',
    'hep-ex': 'High Energy Physics - Experiment',
    'hep-lat': 'High Energy Physics - Lattice',
    'hep-ph': 'High Energy Physics - Phenomenology',
    'hep-th': 'High Energy Physics - Theory',
    'math.AC': 'Commutative Algebra',
    'math.AG': 'Algebraic Geometry',
    'math.AP': 'Analysis of PDEs',
    'math.AT': 'Algebraic Topology',
    'math.CA': 'Classical Analysis and ODEs',
    'math.CO': 'Combinatorics',
    'math.CT': 'Category Theory',
    'math.CV': 'Complex Variables',
    'math.DG': 'Differential Geometry',
    'math.DS': 'Dynamical Systems',
    'math.FA': 'Functional Analysis',
    'math.GM': 'General Mathematics',
    'math.GN': 'General Topology',
    'math.GR': 'Group Theory',
    'math.GT': 'Geometric Topology',
    'math.HO': 'History and Overview',
    'math.IT': 'Information Theory',
    'math.KT': 'K-Theory and Homology',
    'math.LO': 'Logic',
    'math.MG': 'Metric Geometry',
    'math.MP': 'Mathematical Physics',
    'math.NA': 'Numerical Analysis',
    'math.NT': 'Number Theory',
    'math.OA': 'Operator Algebras',
    'math.OC': 'Optimization and Control',
    'math.PR': 'Probability',
    'math.QA': 'Quantum Algebra',
    'math.RA': 'Rings and Algebras',
    'math.RT': 'Representation Theory',
    'math.SG': 'Symplectic Geometry',
    'math.SP': 'Spectral Theory',
    'math.ST': 'Statistics Theory',
    'math-ph': 'Mathematical Physics',
    'funct-an': 'Functional Analysis',
    'alg-geom': 'Algebraic Geometry',
    'nlin.AO': 'Adaptation and Self-Organizing Systems',
    'chao-dyn': 'Chaotic Dynamics',
    'nlin.CD': 'Chaotic Dynamics',
    'nlin.CG': 'Cellular Automata and Lattice Gases',
    'nlin.PS': 'Pattern Formation and Solitons',
    'nlin.SI': 'Exactly Solvable and Integrable Systems',
    'nucl-ex': 'Nuclear Experiment',
    'nucl-th': 'Nuclear Theory',
    'physics.acc-ph': 'Accelerator Physics',
    'physics.ao-ph': 'Atmospheric and Oceanic Physics',
    'physics.app-ph': 'Applied Physics',
    'physics.atm-clus': 'Atomic and Molecular Clusters',
    'physics.atom-ph': 'Atomic Physics',
    'physics.bio-ph': 'Biological Physics',
    'physics.chem-ph': 'Chemical Physics',
    'physics.class-ph': 'Classical Physics',
    'physics.comp-ph': 'Computational Physics',
    'physics.data-an': 'Data Analysis, Statistics and Probability',
    'physics.ed-ph': 'Physics Education',
    'physics.flu-dyn': 'Fluid Dynamics',
    'physics.gen-ph': 'General Physics',
    'physics.geo-ph': 'Geophysics',
    'physics.hist-ph': 'History and Philosophy of Physics',
    'physics.ins-det': 'Instrumentation and Detectors',
    'physics.med-ph': 'Medical Physics',
    'physics.optics': 'Optics',
    'physics.plasm-ph': 'Plasma Physics',
    'physics.pop-ph': 'Popular Physics',
    'physics.soc-ph': 'Physics and Society',
    'physics.space-ph': 'Space Physics',
    'q-bio.BM': 'Biomolecules',
    'q-bio.CB': 'Cell Behavior',
    'q-bio.GN': 'Genomics',
    'q-bio.MN': 'Molecular Networks',
    'q-bio.NC': 'Neurons and Cognition',
    'q-bio.OT': 'Other Quantitative Biology',
    'q-bio.PE': 'Populations and Evolution',
    'q-bio.QM': 'Quantitative Methods',
    'q-bio.SC': 'Subcellular Processes',
    'q-bio.TO': 'Tissues and Organs',
    'q-fin.CP': 'Computational Finance',
    'q-fin.EC': 'Economics',
    'q-fin.GN': 'General Finance',
    'q-fin.MF': 'Mathematical Finance',
    'q-fin.PM': 'Portfolio Management',
    'q-fin.PR': 'Pricing of Securities',
    'q-fin.RM': 'Risk Management',
    'q-fin.ST': 'Statistical Finance',
    'q-fin.TR': 'Trading and Market Microstructure',
    'quant-ph': 'Quantum Physics',
    'q-alg' : 'Quantum Algebra',
    'stat.AP': 'Applications',
    'stat.CO': 'Computation',
    'stat.ME': 'Methodology',
    'stat.ML': 'Machine Learning',
    'stat.OT': 'Other Statistics',
    'stat.TH': 'Statistics Theory'}


def msc_tags():
    with open('./data/msc.json','r') as file:
        text = file.read()
        return json.loads(text)

def msc_to_eng(msc_list):
    out = []
    if msc_list is None:
        return None
    for tag in msc_list:
        if tag not in util.msc_tags().keys():
            continue
        else:
            out.append(util.msc_tags()[tag])
        return out




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
        

def find_hyph(text):
    pattern = r'(?<!-)\b(?:\w+)(?=-)(?:-(?=\w)\w+)+(?!-)\b'
    keywords = regex.findall(pattern,text)

    if keywords == []:
        return None
    else:
        return list(set(keywords))

def find_msc(cat_list):
    pattern = r'\b\d{2}[0-9a-zA-Z]{3}\b'
    out = []
    for cat in cat_list:
        tags = regex.findall(pattern,cat)
        for tag in tags:
            out.append(tag)
    return out


def format_query(author='',title='',cat='',abstract=''):
    """Returns a formatted arxiv query string to handle simple queries of at most one instance each of these fields. To leave a field unspecified,
    leave the corresponding argument blank.
    
    e.g. format_query(cat='math.AP') will return the string used to pull all articles with the subject tag 'PDEs'.

    Args:
        author: string to search for in the author field.
        title: string to search for in the title field.
        cat: A valid arxiv subject tag. See the full list of these at:
        https://arxiv.org/category_taxonomy
        abstract: string to search for in the abstract field.

    Returns:
        properly formatted query string to return all results simultaneously matching all specified fields.
    """

    tags = [f'au:{author}', f'ti:{title}', f'cat:{cat}', f'abs:{abstract}'] 
    query = ' AND '.join([tag for tag in tags if not tag.endswith(':')])
    return query



def query_to_df(query,max_results):
    """Returns the results of an arxiv API query in a pandas dataframe.

    Args:
        query: string defining an arxiv query formatted according to 
        https://info.arxiv.org/help/api/user-manual.html#51-details-of-query-construction
        
        max_results: positive integer specifying the maximum number of results returned.

    Returns:
        pandas dataframe with one column for indivial piece of metadata of a returned result.
        To see a list of these columns and their descriptions, see the documentation for the Results class of the arxiv package here:
        http://lukasschwab.me/arxiv.py/index.html#Result

        The 'links' column is dropped and the authors column is a list of each author's name as a string.
        The categories column is also a list of all tags appearing.
    """
    client = arxiv.Client(page_size=100,num_retries=3)
    search = arxiv.Search(
            query = query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.LastUpdatedDate
            )
    results = client.results(search)

    drop_cols = ['authors','links','_raw']
    df = pd.DataFrame()

    for result in results:
        row_dict = {k : v for (k,v) in vars(result).items() if k not in drop_cols}
        row_dict['authors'] = [author.name for author in result.authors]
        row_dict['links'] = [link.href for link in result.links]
        row = pd.Series(row_dict)
        df = pd.concat([df , row.to_frame().transpose()], axis = 0)

    return df.reset_index(drop=True,inplace=False)
