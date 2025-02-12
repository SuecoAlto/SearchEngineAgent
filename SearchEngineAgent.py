# Building an agent that can interact with a search engine. 
# This will enable asking this agent questions, watch it call the search tool, and have conversations with it.
# Agents are systems that use LLMs as reasoning engines to determine which actions to take and the inputs necessary
# to perform the action. After executing actions, the results can be fed back into the LLM to determine whether more
# actions are needed, or whether it is okay to finish. This is often achieved via tool-calling.

# As these applications get more and more complex, it becomes crucial to be able to inspect what exactly is going on 
# inside your chain or agent. The best way to do this is with LangSmith.

from dotenv import load_dotenv
import os
from langchain_community.tools.tavily_search import TavilySearchResults # TavilySearchTool

load_dotenv()

LangSmith_Api_Key = os.getenv("LANG_SMITH_API_KEY")
LangSmith_Tracing = os.getenv("LANGSMITH_TRACING")
Tivaly_Api_Key = os.getenv("TIVALY_API_KEY")

# Initialize LangSmith tracing here
if LangSmith_tracing == "true":
    print("LangSmith tracing is enabled.")

# TavilySearchTool
Search = TavilySearchResults(max_results=5)
UserInput = input("What would you like to search for? ")
SearchResults = Search.invoke(UserInput)
print(SearchResults)
# If we want, we can create other tools. Once we have all the tools we want, 
# we can put them in a list that we will reference later.
Tools = [Search]


