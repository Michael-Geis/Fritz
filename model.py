import pandas as pd
from sklearn.pipeline import Pipeline
from storage import Fetch
from cleaning import TextCleaner
from embedding import Embedder
from search import Search


def get_recs(id_list, save_recs=False):
    path_to_library = "./data/libraries/APSP_50_allenai-specter"
    path_to_save_recs = "./output/"

    ## Create pipeline

    model = Pipeline(
        [
            ("fetch", Fetch()),
            ("clean", TextCleaner()),
            ("embed", Embedder(model_name="allenai-specter")),
            ("search", Search(path_to_library=path_to_library)),
        ]
    )

    recommendation_df = model.transform(id_list)

    if save_recs:
        recommendation_df.to_feather(path_to_save_recs)

    return recommendation_df
