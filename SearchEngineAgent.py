# Building an agent that can interact with a search engine. 
# This will enable asking this agent questions, watch it call the search tool, and have conversations with it.
# Agents are systems that use LLMs as reasoning engines to determine which actions to take and the inputs necessary
# to perform the action. After executing actions, the results can be fed back into the LLM to determine whether more
# actions are needed, or whether it is okay to finish. This is often achieved via tool-calling.

# As these applications get more and more complex, it becomes crucial to be able to inspect what exactly is going on 
# inside your chain or agent. The best way to do this is with LangSmith.

from dotenv import load_dotenv
import os
import getpass # For the llm api key, if not set corectly in the .env file, the user will be prompted to enter it in the terminal
import json

# Tools and models
from langchain_community.tools.tavily_search import TavilySearchResults # TavilySearchTool
from langchain.chat_models import init_chat_model # For the chat model
from langchain_core.messages import HumanMessage # For the human message in LLM
from langgraph.prebuilt import create_react_agent # For the creation of an agent

load_dotenv()

# Api Keys
LangSmith_Tracing = os.getenv("LANGSMITH_TRACING")
LangSmith_Api_Key = os.getenv("LANGSMITH_API_KEY")
Tavily_Api_Key = os.getenv("TAVILY_API_KEY")
OpenAI_Api_Key = os.getenv("OPENAI_API_KEY")

# Error handling for the api keys
if not LangSmith_Api_Key:
    raise ValueError("LANG_SMITH_API_KEY is not set in the environment variables.")
if not Tavily_Api_Key:
    raise ValueError("TAVILY_API_KEY is not set in the environment variables.")

# If the OpenAI API key is not set in the environment variables, the user will be prompted to enter it
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

# If LangSmith tracing is enabled, print a message
if LangSmith_Tracing == "true":
	print("LangSmith tracing is enabled.\n")
else:
	print("LangSmith tracing is disabled.\n")

# TavilySearchTool
try:
    Search_Tool = TavilySearchResults(Api_Key=Tavily_Api_Key, max_results=2)
    User_Search_Input = input("What would you like to search for? ")
    # Ececute the search
    Search_Respons = Search_Tool.invoke(User_Search_Input)
    # Print the search results readably 
    print("\n Search Results: \n")
    print(json.dumps(Search_Respons, indent=2, sort_keys=True))
except Exception as e:
    print(f"An error occurred: {e}")
# If we want, we can create other tools. we can put them in a list that we will reference later.
Tools = [Search_Tool]

#Initializing Language Models
Model = init_chat_model("gpt-3.5-turbo", model_provider="openai")

# Now we can use the model to interact with the tools
# This is done by binding the tools to the model
Model_With_Tools = Model.bind_tools(Tools)
# The user search input is the content of the human message
# contetnt is how we interact with the model, HumanMessage enables human-like interaction
Response = Model_With_Tools.invoke([HumanMessage(content=User_Search_Input)])
# print("\nLLM Response (with tools binding):")
# print(Response.content)


#Create and run the the agent using LangGraph
Agent_Executor = create_react_agent(Model, Tools)

# Run the agent
Agent_Response = Agent_Executor.invoke({"messages": [HumanMessage(content=User_Search_Input)]})

# Print the agent response in raw form
# print("\nAgent Response:")
# print(Agent_Response)

# Print the agent response in a human-readable form
response_text = ""
if isinstance(Agent_Response, dict) and "messages" in Agent_Response:
        for message in Agent_Response["messages"]:
            # Vi antar att AI-meddelandet är av typen AIMessage
            if message.__class__.__name__ == "AIMessage":
                response_text += message.content + "\n"

print("\n Agent:\n", response_text)

