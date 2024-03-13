import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import (
    VectorStoreIndex,
    ServiceContext,
    Document,
    SimpleDirectoryReader,
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
            "content": "Ask me a question about the lectures!",
        }
    ]


# Cache results for improved performance
@st.cache_resource(show_spinner=False)
def load_data():
    # Display a spinner message for ongoing data loading
    with st.spinner(
        text="Loading lecture notes – hang tight! This should take 1-2 minutes."
    ):
        # Load lecture notes using SimpleDirectoryReader from "./data" recursively
        docs = SimpleDirectoryReader(input_dir="./data", recursive=True).load_data()
        # Set up a ServiceContext with default settings and OpenAI model for responses
        service_context = ServiceContext.from_defaults(
            llm=OpenAI(
                model="gpt-3.5-turbo",
                temperature=0.5,
                system_prompt="You are an expert on lecture notes, answering questions for exams. Keep responses technical and factual, avoiding hallucinations.",
            )
        )
        # Create a VectorStoreIndex for efficient organization and indexing
        return VectorStoreIndex.from_documents(docs, service_context=service_context)


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
