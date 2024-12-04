import streamlit as st

from app.constant import MARKDOWN_DIR, DOCUMENT_DIR, EMBEDDINGS_DIR, WEAVIATE_COLLECTION_NAME


st.title("Documents")

st.header("Add Document")
url_input = st.text_input("Wikipedia URL")
if st.button("Add",use_container_width=True,type="primary"):
    st.write("Document added")

