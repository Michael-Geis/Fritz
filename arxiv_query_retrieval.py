import arxiv
import pandas as pd

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
