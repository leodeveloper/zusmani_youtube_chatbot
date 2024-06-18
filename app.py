import streamlit as st
import json
from datetime import datetime
import os

from llama_index.llms.groq import Groq
from llama_index.core import Settings

from llama_index.llms.openai import OpenAI
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

from llama_index.core.response.pprint_utils import pprint_response

from dotenv import load_dotenv
load_dotenv()

os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
llm = Groq(temperature=0,model="llama3-70b-8192")
Settings.llm = llm

# check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("youtubetranscript/text/").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# Function to sort by publish_date
def sort_by_publish_date(item):
    return datetime.strptime(item['publish_date'], '%Y-%m-%d %H:%M:%S')

# Function to load video titles from JSON file
def load_video_titles():
    with open('extracted_data.json', 'r') as f:
        data = json.load(f)
        # Sorting the JSON array by publish_date
        sorted_json_array = sorted(data, key=sort_by_publish_date, reverse=True)
    return sorted_json_array

# Function to simulate response from LLM model (replace with actual model logic)
def generate_response(user_input):
    query_engine = index.as_chat_engine(chat_mode="context", llm=llm,verbose=True)
    response = query_engine.chat(user_input)
    return response.response

# Sidebar content
st.sidebar.title('Zeeshan Usmani YouTube Channel')
st.sidebar.image('https://yt3.googleusercontent.com/a-Sr9AQV5fW43HXMCV7FmFOb5ngJx3_jm7lsFf2q0MyM3-RUMQbW9Sa-2o8AizzGUjMRLO-wPQ=s160-c-k-c0x00ffffff-no-rj', use_column_width=True)
#st.sidebar.header("YouTube Video's")
#video_titles = load_video_titles()
#for item in video_titles:
    #st.sidebar.image(f"{item['thumbnail_url']}", use_column_width=True)
#    st.sidebar.write(f"{item['title']}")
#    st.sidebar.write(f"{item['publish_date']}")

# Main content
st.title('Zeeshan Usmani YouTube Channel Chatbot')
st.header('You may chat with Zeeshan Usmani YouTube Channel videos')
user_input = st.text_area('Enter your message:')
if st.button('Show Response'):
    response = generate_response(user_input)
    #st.write_stream(response)
    st.write(response)

# Footer
st.markdown('---')
st.write('All copyrights are reserved 2024. For more information, contact [leodeveloper@gmail.com](mailto:leodeveloper@gmail.com).')
