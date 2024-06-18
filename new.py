import json
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()
from llama_index.llms.groq import Groq

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    PromptTemplate
)

template = (
    "You are a knowledgeable assistant specializing in providing information about YouTube channel videos. \n"
    "When a user asks a question, you will provide a detailed answer along with relevant information about related YouTube videos.\n"
    " Your response should be formatted as a JSON object with the following structure \n"
    "please answer the question: {question}\n"
    "Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \n"
    "{response_json}"
)
qa_template = PromptTemplate(template)

#loading json file
with open('response.json','r') as file:
    RESPONSE_JSON = json.load(file)

# you can create text prompt (for completion API)
prompt = qa_template.format(question="give me the summary of latest video of zeeshan usmani", response_json=RESPONSE_JSON)

print(prompt)

os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
llm = Groq(temperature=1,model="llama3-70b-8192")
#Settings.llm = llm


PERSIST_DIR = "./storage"

storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR, prompt_helper=qa_template)
index = load_index_from_storage(storage_context)

query_engine = index.as_chat_engine(chat_mode="context", llm=llm,verbose=True)
response = query_engine.chat({question:"give me the summary of latest video of zeeshan usmani",response_json:RESPONSE_JSON})
print(response.response)
# Store the formatted prompt in the StorageContext
#storage_context.store('formatted_prompt', prompt)

# Retrieve and print the stored prompt
#retrieved_prompt = storage_context.retrieve('formatted_prompt')
#print(retrieved_prompt)



