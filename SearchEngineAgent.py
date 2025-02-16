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
import asyncio

# Tools and models
from langchain_community.tools.tavily_search import TavilySearchResults # TavilySearchTool
from langchain.chat_models import init_chat_model # For the chat model
from langchain_core.messages import HumanMessage, AIMessage # For the human message in LLM
from langgraph.prebuilt import create_react_agent # For the creation of an agent
from langgraph.checkpoint.memory import MemorySaver # For saving the agent state in memory


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
# If the OpenAI API key is not set in the environment variables, the user will  beprompted to enter it
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

# If LangSmith tracing is enabled, print a message
if LangSmith_Tracing == "true":
    print("LangSmith tracing is enabled.\n")
else:
    print("LangSmith tracing is disabled.\n")

# TavilySearchTool
Search_Tool = TavilySearchResults(tavily_api_key=Tavily_Api_Key, max_results=2)
Model = init_chat_model("gpt-3.5-turbo", model_provider="openai")

# If we want, we can create other tools. we can put them in a list that we will reference later.
Tools = [Search_Tool]

# Now we can use the model to interact with the tools
# This is done by binding the tools to the model
Model_With_Tools = Model.bind_tools(Tools)

# Create agent memory
Memory = MemorySaver()

# Create and run the agent using LangGraph
Agent_Executor = create_react_agent(Model, Tools, checkpointer=Memory)

config = {
    "run_name": "SEARCH_AGENT",
    "configurable": {
    "thread_id": "abc123",
    "checkpoint_ns":"namespace",
    "checkpoint_id":"checkpoint1"}
    }

Search_History = []

User_Search_Input = input("What would you like to search for? ")
Search_History.append(User_Search_Input)

# Execute the search
try:
    Search_Respons = Search_Tool.invoke(User_Search_Input)

    # Print the search results readably 
    print("\n\n***Search Results***\n")
    print(json.dumps(Search_Respons, indent=2, sort_keys=True))
    print("------------------------ \n")

except Exception as e:
    print(f"An error occurred: {e}")
    exit()
    

print("\n***The message stream as they occur***\n")

for Chunk in Agent_Executor.stream(
    {"messages": [HumanMessage(content=User_Search_Input)]}, config=config
):
    print(Chunk)
    print("------------------------ \n")

#Token Stream
# Stream the agent response as tokens for real-time interaction - The stream agent respons line by line as it occurs

print("\n***The token stream as they occur***\n")

async def stream_tokens():

    async for Event in Agent_Executor.astream_events(
        {"messages": [HumanMessage(content=User_Search_Input)]}, version="v1", config=config
    ):
        Kind = Event["event"]
        
        if Kind == "on_chain_start":
            if (Event["name"] == "agent"):
                print(f"Starting agent {Event['name']} with input {Event ['data'].get('input')}, \n\nTokens:\n")

        elif Kind == "on_chain_end":
            if (Event["name"] == "agent"):
                output_data = Event["data"].get("output", {})
                print(f"Done agent: {Event['name']} with output: {output_data.get('output', 'No output available')}")

        elif Kind == "on_chat_model_stream":
            content = Event["data"]["chunk"].content
            if content:
                print(content, end="|")
        
        elif Kind == "on_tool_start":
            print("\n-------------2-----------\n")
            print(f"Starting tool: {Event['name']} with inputs: {Event['data'].get('input')}")

        elif Kind == "on_tool_end":
            print(f"\nDone tool: {Event['name']}")
            print("\n-------------3-----------\n")
            print(f"Tool output was:\n{Event['data'].get('output')}")

print("-------------1-----------\n")

asyncio.run(stream_tokens())
# Initializing Language Models

# Run the agent
Agent_Response = Agent_Executor.invoke({"messages": [HumanMessage(content=str(Search_Respons))]}, config=config)

# Continuous chat loop with the LLM (Agent memory is retained)
# Print response
Response_Text = ""
if isinstance(Agent_Response, dict) and "messages" in Agent_Response:
    for message in Agent_Response["messages"]:
        if isinstance(message, AIMessage):
            Response_Text += message.content


# The user search input is the content of the human message
# content is how we interact with the model, HumanMessage enables human-like interaction

#######
print("\n\n***Agent+LLM Initial Response***\n", Response_Text)

# Add user message to conversation history
Conversation_History = [
    HumanMessage(content=User_Search_Input),
    AIMessage(content=Response_Text)
    ]



print("\nYou can now chat with the LLM. Type 'exit' to stop.")

while True:
    user_input = input("\nYou: ")
    
    if user_input.lower() in ["exit", "quit"]:
        print("Ending conversation...")
        break

    Conversation_History.append(HumanMessage(content=user_input))

    if "search" in user_input.lower() and "before" in user_input.lower():
        Response_Text = f"Your previous searches were: {', '.join(Search_History)}"
    else:
        #  **Kör agenten med uppdaterad historik**
        Agent_Response = Agent_Executor.invoke({"messages": Conversation_History}, config=config)

        #  **Hämta och spara svaret från LLM**
        Response_Text = ""
        if isinstance(Agent_Response, dict) and "messages" in Agent_Response:
            for message in Agent_Response["messages"]:
                if isinstance(message, AIMessage):
                    Response_Text = message.content


    # **Spara svaret i konversationshistoriken**
    Conversation_History.append(AIMessage(content=Response_Text))

    # **Skriv ut LLM:s svar**
    print("\nAgent:", Response_Text)


