import os

os.environ["OPENAI_API_KEY"] = ""
os.environ["SERPAPI_API_KEY"] = ""

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
    


llm = OpenAI(temperature=0)

tools = load_tools(["serpapi"], llm=llm)

