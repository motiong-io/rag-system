from openai import OpenAI
import streamlit as st
from ui.st_utils.constant import ST_HIDE_HEADER_HTML
st.markdown(ST_HIDE_HEADER_HTML, unsafe_allow_html=True)


from app.config import env

openai_api_key=env.openai_api_key
openai_base_url=env.openai_base_url
client = OpenAI(api_key=openai_api_key,base_url=openai_base_url)


st.title("Chatbot",anchor=False)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})



