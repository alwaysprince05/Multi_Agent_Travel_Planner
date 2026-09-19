# ✈️ Multi-Agent Travel Planner AI

A sophisticated multi-agent AI system that creates comprehensive, personalized travel plans through the collaboration of 4 specialized agents. Built with **LangGraph** for workflow orchestration and **LangChain** for LLM integration.

## 🌐 Live Demo

**🚀 [View Live App](https://multiagenttravelplannergit-5fj4hxehklqqcgbaeepour.streamlit.app/)** — deployed on Streamlit Community Cloud.

Enter a travel request like *"I want to visit Tokyo for 5 days with a budget of $2000. I love food and temples."* and watch the 4 agents collaborate in real time.

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
- **LLM Engine**: [Groq Cloud](https://console.groq.com) (GPT-OSS-120B)
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

## 🧪 Testing

Run the offline test suite (no API key needed — verifies imports, the LangGraph workflow, and error handling):

```bash
python test_system.py
```

Run the full end-to-end test (makes 4 real Groq API calls, requires `GROQ_API_KEY` in `.env`):

```bash
python test_system.py --live
```

## ☁️ Deployment (Streamlit Community Cloud — Free)

1. Push this repository to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app** → select your repo and branch.
4. Main file path: `multi_agent_system_streamlit.py`
5. Open **Advanced settings → Secrets** and add:

   ```toml
   GROQ_API_KEY = "your_gsk_key_here"
   ```

6. Click **Deploy**. Streamlit auto-installs `requirements.txt` and the app reads the key from `st.secrets` automatically.

> The app looks for `GROQ_API_KEY` in the environment first (local `.env`) and falls back to Streamlit Secrets (cloud), so the same file works in both places. Never commit your real `.env` file — it is already excluded via `.gitignore`.

## 👤 Author

**Prince Maurya**

## 🔗 Submission Links

- **GitHub Repository:** https://github.com/alwaysprince05/Multi_Agent_Travel_Planner
- **Live Demo (Streamlit Cloud):** https://multiagenttravelplannergit-5fj4hxehklqqcgbaeepour.streamlit.app/
- **Demo Video (Google Drive):** [Add your video link here]

## 🙏 Acknowledgments

- The LangChain and LangGraph teams for the state-of-the-art AI frameworks.
- Groq for the ultra-fast LLM inference.
- Streamlit for the beautiful web UI components.
