import pandas as pd
from sklearn.pipeline import Pipeline
from src.storage import Fetch
from src.cleaning import TextCleaner
from src.embedding import Embedder
from src.search import Search


def main(id_list, save_recs=False):
    path_to_library = "./data/libraries/APSP_50_allenai-specter"
    path_to_save_recs = ""

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


if __name__ == "main":
    main()
