import streamlit as st

from app.constant import MARKDOWN_DIR, DOCUMENT_DIR, EMBEDDINGS_DIR, WEAVIATE_COLLECTION_NAME

from ui.st_utils.constant import ST_HIDE_HEADER_HTML
st.markdown(ST_HIDE_HEADER_HTML, unsafe_allow_html=True)


st.title("Documents")



