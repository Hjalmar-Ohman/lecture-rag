import os
import streamlit as st
import openai
import nltk

# Set up NLTK to use the local `nltk_data` folder
nltk_data_dir = "./nltk_data"
nltk.data.path.append(nltk_data_dir)

# Proceed to import llama_index now that the path is set
from llama_index.llms.openai import OpenAI
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings
)

st.set_page_config(
    page_title="Chat with the lecture notes",
    page_icon="📃",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

openai.api_key = st.secrets.openai_key

st.header("Chat with your lecture notes! 💬 📚")

# Initialize the chat messages history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about the lectures! For example 'What is a Ripple-Carry adder?'",
        }
    ]


# Cache results for improved performance
@st.cache_resource(show_spinner=False)
def load_data():
    # Display a spinner message for ongoing data loading
    with st.spinner("Loading lecture notes – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./lecture slides", recursive=True)
        docs = reader.load_data()
        Settings.llm = OpenAI(
            model="gpt-4o-mini",
            temperature=0.5,
            system_prompt="""You are an expert on lecture notes, answering questions for exams. 
            Keep responses technical and factual, avoiding hallucinations. 
            Always be specific to what document and page I can read more about the topic, 
            for example 'This is discussed in Lecture_1_Slides.pdf page 5'""",
        )
        index = VectorStoreIndex.from_documents(docs)
        return index


index = load_data()

# Initialize the chat engine
if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True
    )

# Prompt for user input and save to chat history
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display the prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If the last message is not from the assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Get the response from the chat engine based on the user's input (prompt)
            response = st.session_state.chat_engine.chat(prompt)

            # Display the chatbot's response in the UI using st.write
            st.write(response.response)

            # Create a message object for the chatbot's response
            message = {"role": "assistant", "content": response.response}

            # Add the chatbot's response to the message history
            st.session_state.messages.append(message)  # Add response to message history
