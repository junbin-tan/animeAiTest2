import os
import sys
import constants
import pandas as pd

from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_csv_agent
from IPython.display import display, Markdown
import dotenv


os.environ["OPENAI_API_KEY"] = constants.APIKEY

df = pd.read_csv("anime-filtered_1921.csv")

defaultAgent = create_csv_agent (
    OpenAI(temperature=0),
    "anime-filtered_1921.csv",
    verbose=True
)

gpt4Agent = create_csv_agent (
    ChatOpenAI(temperature=0, model_name="gpt-4"),
    "anime-filtered_1921.csv",
    verbose=True
)

#checking what prompts are currently supplied to model
# print(defaultAgent.agent.llm_chain.prompt.template)