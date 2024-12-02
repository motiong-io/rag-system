import streamlit as st
pages = {
    "Chatbot": [
        st.Page("sub_app/chatbot.py", title="Chatbot"),
        st.Page("sub_app/documents.py", title="Documents"),
    ],
    "Settings": [
        st.Page("sub_app/settings.py", title="Settings"),
    ],
}

pg = st.navigation(pages)
pg.run()