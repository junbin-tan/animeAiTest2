import os
import sys
import constants
import pandas as pd

from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_csv_agent
import dotenv

from langchain.document_loaders import CSVLoader



os.environ["OPENAI_API_KEY"] = constants.APIKEY

# TRAIL 1 : FAILED DOES NOT WORK WELL

# df = pd.read_csv("anime-filtered_1921.csv")

# defaultAgent = create_csv_agent (
#     OpenAI(temperature=0),
#     "anime-filtered_1921.csv",
#     verbose=True
# )

# gpt3_5Agent = create_csv_agent (
#     ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k"),
#     "anime-filtered_1921.csv",
#     verbose=True
# )

#checking what prompts are currently supplied to model
# print(defaultAgent.agent.llm_chain.prompt.template)

# question = input("What question do you have? ")

# print(gpt3_5Agent.run(question))


# TRAIL 2: 

#load document 
loader = CSVLoader("anime-filtered_1921.csv")

#create index
index = VectorstoreIndexCreator()
document = index.from_loaders([loader])

#create question and answering chain using index
chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=document.vectorstore.as_retriever(),  input_key="question")

question = input("What is the question? ")
response = chain({"question" : question})

print(response['result'])