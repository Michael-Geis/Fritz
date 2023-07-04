import arxiv
import pandas as pd
import data_cleaning as clean
from sklearn.preprocessing import MultiLabelBinarizer
import os


class ArXivData:
    """A light class for storing the metadata of a collection of arXiv papers."""

    def __init__(self):
        self.metadata = None
        self.arxiv_subjects = None
        self._returned_metadata = None

    def load_from_feather(self, dataset_file_name, path_to_data_dir):
        path_to_dataset = os.path.join(path_to_data_dir, dataset_file_name)
        self._returned_metadata = pd.read_feather(path_to_dataset)

        self.metadata = self._returned_metadata.drop(columns=["arxiv_subjects"])
        self.arxiv_subjects = self.get_OHE_arxiv_subjects(self._returned_metadata)

    def load_from_query(self, query, max_results, offset=0, raw=False):
        if raw:
            self._returned_metadata = query_to_df(
                query=query, max_results=max_results, offset=offset, raw=True
            )

        else:
            self._returned_metadata = query_to_df(
                query=query, max_results=max_results, offset=offset
            )

            self.metadata = self._returned_metadata.drop(columns="arxiv_subjects")
            self.arxiv_subjects = self.get_OHE_arxiv_subjects(self._returned_metadata)

    def save_as_feather(self, dataset_file_name, path_to_data_dir):
        if self._returned_metadata is None:
            raise Exception(
                "No data stored. Run load_from_query or load_from_feather to retrieve data."
            )

        path_to_dataset = os.path.join(path_to_data_dir, dataset_file_name)
        self._returned_metadata.to_feather(path_to_dataset)

    def get_OHE_arxiv_subjects(self, returned_metadata):
        mlb = MultiLabelBinarizer()

        OHE_arxiv_subjects_array = mlb.fit_transform(returned_metadata.arxiv_subjects)
        arxiv_subject_labels = clean.category_map()

        OHE_arxiv_subjects = pd.DataFrame(
            OHE_arxiv_subjects_array, columns=mlb.classes_
        ).rename(columns=arxiv_subject_labels)

        ## Remove duplicated columns
        return OHE_arxiv_subjects.loc[
            :, ~OHE_arxiv_subjects.columns.duplicated()
        ].copy()


def format_query(author="", title="", cat="", abstract=""):
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

    tags = [f"au:{author}", f"ti:{title}", f"cat:{cat}", f"abs:{abstract}"]
    query = " AND ".join([tag for tag in tags if not tag.endswith(":")])
    return query


def query_to_df(query, max_results, offset, raw=False):
    """Returns the results of an arxiv API query in a pandas dataframe.

    Args:
        query: string defining an arxiv query formatted according to
        https://info.arxiv.org/help/api/user-manual.html#51-details-of-query-construction

        max_results: positive integer specifying the maximum number of results returned.

        chunksize:

    Returns:
        pandas dataframe with one column for indivial piece of metadata of a returned result.
        To see a list of these columns and their descriptions, see the documentation for the Results class of the arxiv package here:
        http://lukasschwab.me/arxiv.py/index.html#Result

        The 'links' column is dropped and the authors column is a list of each author's name as a string.
        The categories column is also a list of all tags appearing.
    """
    client = arxiv.Client(page_size=2000, num_retries=3)
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

    raw_metadata = pd.DataFrame(metadata_generator, columns=columns, index=index)

    returned_metadata = raw_metadata.copy().drop(columns=["categories"])
    returned_metadata["arxiv_subjects"] = clean.extract_arxiv_subjects(raw_metadata)
    returned_metadata["msc_tags"] = clean.extract_msc_tags(raw_metadata)

    if raw:
        return raw_metadata

    return returned_metadata
