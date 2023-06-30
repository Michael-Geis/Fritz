import arxiv
import pandas as pd
import data_cleaning as clean
from sklearn.preprocessing import MultiLabelBinarizer

class ArXivData():
    """A light class for storing the metadata of a collection of arXiv papers.
    """   

    def __init__(self):
        """
        data: dataframe holding the metadata. Each row represents a paper and each column is
        a separate piece of metadata.
        
        query: A tuple of the form (query_string,max_results) where query_string is the formatted 
        string that produced the raw data and max_results is the value of that parameter passed to the
        arXiv API.

        raw: The original, raw dataset as returned by the arXiv API, if current data is clean.

        cats: A DataFrame containing one-hot-encoded categories of the self.data DataFrame.
        """

        self.data = None
        self.query = None
        self.categories = None

    def load_from_file():
        pass    

    def load_from_query(self,query_string,max_results,offset):
        self.data = query_to_df(query=query_string,max_results=max_results,offset=offset)        
        self.query = (query_string,max_results)
        #self.categories = self.get_OHE_cats()
        
        
    def clean(self,dataset):
        """Constructs this dataset by cleaning another one.

        Args:
            dataset: An ArXivData object containing data to be cleaned.
        """
        self.data = clean.clean(dataset)
        self.query = dataset.query
        self.raw = dataset.raw
        self.categories = dataset.categories

    def get_OHE_cats(self):
        mlb = MultiLabelBinarizer()
        OHE_category_array = mlb.fit_transform(self.data.categories)
        return pd.DataFrame(
            OHE_category_array, columns = mlb.classes_).rename(
            mapper=clean.category_map())





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



def query_to_df(query,max_results,offset):
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
    client = arxiv.Client(page_size=2000,num_retries=3)
    search = arxiv.Search(
            query = query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.LastUpdatedDate
            )
    
    columns = ['title','summary','categories','id']
    index = range(offset,max_results)


    results = client.results(search,offset=offset)

    metadata_generator = ((result.title,result.summary,    
                        result.categories,
                        result.entry_id.split('/')[-1]) for result in results)
    
    metadata_dataframe = pd.DataFrame(metadata_generator, columns=columns, index=index)


    return metadata_dataframe







