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
import getpass # For the llm api key, if not set corectly in the .env file, the user will be prompted to enter it in the terminal
from langchain.chat_models import init_chat_model # For the chat model
from langchain_core.messages import HumanMessage # For the human message in LLM

load_dotenv()

LangSmith_Tracing = os.getenv("LANGSMITH_TRACING")
LangSmith_Api_Key = os.getenv("LANGSMITH_API_KEY")
Tavily_Api_Key = os.getenv("TAVILY_API_KEY")
OpenAI_Api_Key = os.getenv("OPENAI_API_KEY")

# Kontrollera att API-nycklarna är laddade korrekt
if not LangSmith_Api_Key:
    raise ValueError("LANG_SMITH_API_KEY is not set in the environment variables.")
if not Tavily_Api_Key:
    raise ValueError("TIVALY_API_KEY is not set in the environment variables.")
if not OpenAI_Api_Key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

# Initialize LangSmith tracing here
if LangSmith_Tracing == "true":
    print("LangSmith tracing is enabled.\n")

try:
    # TavilySearchTool
    Search = TavilySearchResults(Api_Key=Tavily_Api_Key, max_results=2)
    UserInput = input("What would you like to search for? ")
    SearchResults = Search.invoke(UserInput)
    print("\n",SearchResults)
except Exception as e:
    print(f"An error occurred: {e}")
# If we want, we can create other tools. Once we have all the tools we want, 
# we can put them in a list that we will reference later.
Tools = [Search]

#Initializing Language Models
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

model = init_chat_model("gpt-3.5-turbo", model_provider="openai")
	
Response = model.invoke([HumanMessage(content="Hi!")])
Response.content
print(Response.content)



