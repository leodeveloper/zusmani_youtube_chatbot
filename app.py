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

#from llama_index.core.response.pprint_utils import pprint_response
#from llama_index.core.memory import ChatMemoryBuffer

from dotenv import load_dotenv
load_dotenv()

os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
llm = Groq(temperature=1,model="llama3-70b-8192")
Settings.llm = llm



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

    query_engine = index.as_chat_engine(chat_mode="context", llm=llm,verbose=True)

   

    response = query_engine.chat(f"{user_input} with yotube link, publish date and view count")
    return response.response

# Sidebar content
st.sidebar.title('Zeeshan Usmani YouTube Channel')
st.sidebar.image('zusmani_channels4_profile.jpg', use_column_width=True)
st.sidebar.header("YouTube Video's")
video_titles = load_video_titles()
count=1
for item in video_titles:
    #st.sidebar.image(f"{item['thumbnail_url']}", use_column_width=True)
    st.sidebar.write(f"{count}.{item['title']}")
    st.sidebar.write(f"{item['publish_date']}")
    st.sidebar.write("-----------------------")
    count+=1

# Main content
st.title('Zeeshan Usmani YouTube Channel Chatbot')
st.header('You may chat with Zeeshan Usmani YouTube Channel videos')
st.write('last update on 06 June 2024')
user_input = st.text_area('Enter your message:')
if st.button('Show Response'):
    with st.spinner("In progress..."):
        response = generate_response(user_input)
        #st.write_stream(response)
        st.write(response)
        st.write('-----------------------')
        #res=pprint_response(response,show_source=True)
        #st.write(res)

# Footer
st.markdown('---')
st.write('All copyrights are reserved 2024. For more information, contact [leodeveloper@gmail.com](mailto:leodeveloper@gmail.com).')
