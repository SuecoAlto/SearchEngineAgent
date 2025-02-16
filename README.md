# 🔎 AI-Powered Search & Chat Agent

This project implements an **AI-powered search and chat agent** using **LangChain, LangGraph, and Tavily Search API**. The agent can retrieve **real-time search results** and engage in **interactive conversations** with users, providing a seamless and dynamic experience.

---

## 🚀 Features

- **🔍 Search Engine Integration**: Uses **Tavily API** to fetch real-time web search results.
- **💬 Conversational AI**: Chat with an **LLM-powered agent** that remembers the conversation.
- **⏳ Streaming Responses**: **Real-time token streaming** for a more engaging chat experience.
- **🧠 Memory Handling**: Retains conversation history for contextual understanding.
- **📊 LangSmith Tracing**: Debugging and tracing using **LangSmith** (optional).
- **🔧 Tool Integration**: Extensible architecture for adding more tools.

---

## 🛠️ Installation & Setup

1️⃣ Clone the Repository

```sh
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

3️⃣ Set Up Environment Variables

```
Create a .env file and add your API keys:

LANGSMITH_TRACING=true  # Set to false if debugging is not needed
LANGSMITH_API_KEY=your-langsmith-api-key
TAVILY_API_KEY=your-tavily-api-key
OPENAI_API_KEY=your-openai-api-key
```

4️⃣ Run the Chatbot

```
python SearchEngineAgent.py
```

## ⚡ Usage

```
	1.	Start the script and enter your search query.
	2.	The agent fetches search results and returns real-time data.
	3.	Engage in a conversation with the LLM-powered assistant.
	4.	Chat history is stored, and you can ask about previous searches.
	5.	Type "exit" to terminate the session.
```

## 📂 Project Structure

```
📂 project-folder
│-- SearchEngineAgent.py  # Main script
│-- .env                  # Environment variables (ignored in Git)
│-- requirements.txt       # Python dependencies
│-- README.md              # This file
```

---

## 🛠️ Built With

- **LangChain** - Framework for building LLM applications.
- **LangGraph** - Agent execution framework.
- **Tavily Search** - Real-time search API.
- **OpenAI API** - LLM-powered chat agent.
- **Python** - Main programming language.

---

## 🔮 Future Improvements

- **Enhanced Search Capabilities** (support for multiple sources).
- **Integration with More LLMs** (GPT-4, Claude, etc.).
- **Custom User Profiles & Memory Handling**.
- **Web-Based Frontend Interface**.

```

```
