import arxiv
import pandas as pd
import numpy as np
import cleaning as clean
from dataclasses import dataclass, astuple, asdict


class ArXivData:
    """A class for storing the metadata of a collection of arXiv papers."""

    def __init__(self) -> None:
        self._returned_metadata = None
        self.metadata = None
        self.arxiv_subjects = None
        self.doc_strings = "title and abstract"
        self.embeddings = None

    def load_from_feather(self, path_to_dataset):
        """Loads metadata from a saved feather file.

        Args:
            path_to_dataset: path to the feather file containing the dataset.
        """
        self._returned_metadata = pd.read_feather(path_to_dataset)
        self.metadata = self._returned_metadata
        self.arxiv_subjects = clean.OHE_arxiv_subjects(self.metadata)

    def load_from_query(self, query, max_results, offset=0):
        """Loads instance with data returned from an ArXiv API query.

        Args:
            query: query string used to call the API
            max_results: maximum number of results from the API call to return
            offset: number of results to skip over initially. Defaults to 0.
        """

        self._returned_metadata = query_to_df(
            query=query, max_results=max_results, offset=offset
        )
        self.metadata = clean.split_categories(self._returned_metadata)
        self.arxiv_subjects = clean.OHE_arxiv_subjects(self.metadata)

    def load_from_id_list(self, id_list):
        self._returned_metadata = query_to_df(id_list=id_list, max_results=len(id_list))
        self.metadata = clean.split_categories(self._returned_metadata)
        self.arxiv_subjects = clean.OHE_arxiv_subjects(self.metadata)

    def save_as_feather(self, path_to_dataset):
        """Saves a dataset as a feather file.

        Args:
            path_to_dataset: directory to save the dataset

        Raises:
            Exception: Raises exception if there is no data to be saved.
        """

        if self.metadata.empty:
            raise Exception(
                "No data stored. Run load_from_query or load_from_feather to retrieve data."
            )
        self.metadata.to_feather(path_to_dataset)


def query_to_df(query=None, id_list=None, max_results=None, offset=0):
    """Returns the results of an arxiv API query in a pandas dataframe.

    Args:
        query: string defining an arxiv query formatted according to
        https://info.arxiv.org/help/api/user-manual.html#51-details-of-query-construction

        max_results: positive integer specifying the maximum number of results returned.

        id_list: A list of arxiv ids as strings to retrieve

    Returns:
        pandas dataframe with one column for indivial piece of metadata of a returned result.
        To see a list of these columns and their descriptions, see the documentation for the Results class of the arxiv package here:
        http://lukasschwab.me/arxiv.py/index.html#Result

        The 'links' column is dropped and the authors column is a list of each author's name as a string.
        The categories column is also a list of all tags appearing.
    """
    client = arxiv.Client(page_size=2000, num_retries=10)

    if id_list:
        search = arxiv.Search(
            id_list=id_list,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.LastUpdatedDate,
        )

    else:
        if not query:
            raise Exception(
                "You must pass either a query string or a list of arxiv IDs"
            )

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.LastUpdatedDate,
        )

    columns = ["title", "summary", "categories", "id"]
    index = range(offset, max_results)

    results = client.results(search, offset=offset)

    metadata_generator = (
        (
            result.title,
            result.summary,
            result.categories,
            result.entry_id.split("/")[-1],
        )
        for result in results
    )

    returned_metadata = pd.DataFrame(metadata_generator, columns=columns, index=index)
    returned_metadata = returned_metadata.rename(columns={"summary": "abstract"})
    return returned_metadata


# def format_query(author="", title="", cat="", abstract=""):
#     """Returns a formatted arxiv query string to handle simple queries of at most one instance each of these fields. To leave a field unspecified,
#     leave the corresponding argument blank.

#     e.g. format_query(cat='math.AP') will return the string used to pull all articles with the subject tag 'PDEs'.

#     Args:
#         author: string to search for in the author field.
#         title: string to search for in the title field.
#         cat: A valid arxiv subject tag. See the full list of these at:
#         https://arxiv.org/category_taxonomy
#         abstract: string to search for in the abstract field.

#     Returns:
#         properly formatted query string to return all results simultaneously matching all specified fields.
#     """

#     tags = [f"au:{author}", f"ti:{title}", f"cat:{cat}", f"abs:{abstract}"]
#     query = " AND ".join([tag for tag in tags if not tag.endswith(":")])
#     return query
