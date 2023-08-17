import streamlit as st
import arxiv
import pandas as pd
from src.model import get_recs


# Function to extract the details of the paper
def arxiv_search(input_id):
    paper = next(arxiv.Search(id_list=[input_id]).results())
    return paper


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    # Title for the dashboard
    st.title("ArXiv recommender")

    # Input article; currently only one input
    input_arxiv_id = st.text_input("Insert arXiv id here: ")

    if input_arxiv_id:
        # Details of the extracted paper are stored
        input_data = arxiv_search(input_arxiv_id)
        # Dropdown for the input article
        with st.expander("%s" % input_data.title):
            st.write("Abstract: ", input_data.summary)

        if st.button("Show Abstract"):
            st.write("Abstract: ", input_data.summary)

        # Loading the stored corpus and embeddings and topics
        embeddings = pd.read_feather(
            "./data/libraries/APSP_50_allenai-specter/embeddings.feather"
        ).values

        # # Initializing the model
        # model = sentence_transformers.SentenceTransformer("allenai-specter")

        # # Encoding the title and summary of the input article
        # input_embedding = model.encode(input_data.summary)

        # # Top 5 recommendations from the corpus
        # reco = sentence_transformers.util.semantic_search(
        #     query_embeddings=input_embedding, corpus_embeddings=embeddings, top_k=5
        # )

        # reco_id = [recs["corpus_id"] for recs in reco[0]]

        # # Loading the metadata
        # corpus = pd.read_feather(
        #     "./data/libraries/APSP_50_allenai-specter/metadata.feather"
        # )
        recs = get_recs(id_list=[input_arxiv_id])

        st.write("Top 5 similar articles")

        for i in range(5):
            with st.expander("%s" % recs.title.tolist()[i]):
                st.write("Abstract: ", recs.abstract.tolist()[i])

    else:
        pass
