# ✈️ Multi-Agent Travel Planner AI

A sophisticated multi-agent AI system that creates comprehensive, personalized travel plans through the collaboration of 4 specialized agents. Built with **LangGraph** for workflow orchestration and **LangChain** for LLM integration.

## 📋 Overview

This system transforms a simple travel request (e.g., *"I want to visit Tokyo for 5 days with a budget of $2000"*) into a detailed day-by-day itinerary, destination research, and budget analysis. 

## 🤖 Multi-Agent Architecture

The system uses a directed graph workflow where each agent has a specific role:

1.  **Planner Agent**: Extracts structured data (destination, dates, budget, preferences) from natural language.
2.  **Research Agent**: Gathers destination-specific highlights, attractions, and practical travel tips.
3.  **Itinerary Builder Agent**: Synthesizes research and user preferences into a detailed daily schedule.
4.  **Budget Estimator Agent**: Provides a full cost breakdown and assesses the feasibility of the itinerary against the user's budget.

## 🛠️ Technology Stack

- **Workflow**: [LangGraph](https://github.com/langchain-ai/langgraph)
- **AI Framework**: [LangChain](https://github.com/langchain-ai/langchain)
- **LLM Engine**: [Groq Cloud](https://console.groq.com) (Llama-3.3-70b)
- **Interface**: [Streamlit](https://streamlit.io/)
- **Environment**: Python 3.9+

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.9 or higher
- A Groq API Key (Get one for free at [console.groq.com](https://console.groq.com))

### 2. Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd Multi-Agent-Travel-Planner

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the project root and add your API key:
```env
GROQ_API_KEY=your_gsk_key_here
```

### 4. Running the App
```bash
streamlit run multi_agent_system_streamlit.py
```

## 📍 Features

- **Natural Language Understanding**: Just talk to the agents like a human.
- **Progress Tracking**: See the status as each agent completes its task.
- **Tabbed Results**: View Trip Overview, Itinerary, Budget, and Research in separate clean views.
- **Downloadable Plans**: Export your final travel plan to a text file with one click.

## 👤 Author

**Prince Maurya**

## 🙏 Acknowledgments

- The LangChain and LangGraph teams for the state-of-the-art AI frameworks.
- Groq for the ultra-fast LLM inference.
- Streamlit for the beautiful web UI components.
