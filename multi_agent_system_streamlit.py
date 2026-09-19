"""
Multi-Agent Travel Planner System
Assignment: Build a Multi-Agent System using LangChain + LangGraph

This system uses 4 specialized agents to create comprehensive travel plans:
1. Planner Agent - Extracts structured information from user requests
2. Research Agent - Gathers destination information and attractions
3. Itinerary Builder Agent - Creates day-by-day travel itineraries
4. Budget Estimator Agent - Provides cost breakdowns and budget analysis

The agents collaborate through a shared state (TravelState) and are orchestrated
using LangGraph's directed workflow.

Usage:
    CLI Mode: python multi_agent_system_streamlit.py
    Streamlit Mode: streamlit run multi_agent_system_streamlit.py
"""

import os
import sys
import json
from typing import TypedDict
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# ============================================================================
# SHARED STATE DEFINITION
# ============================================================================

class TravelState(TypedDict):
    """
    Shared state passed between all agents in the workflow.
    Each agent reads from and writes to specific fields.
    """
    user_input: str          # Raw free-text from the user
    destination: str         # Extracted by Planner; "unknown" if not found
    travel_dates: str        # Extracted by Planner; empty string if not found
    budget: str              # Extracted by Planner; empty string if not found
    preferences: str         # Extracted by Planner; empty string if not found
    research_notes: str      # Populated by Research Agent
    itinerary: str           # Populated by Itinerary Builder Agent
    budget_estimate: str     # Populated by Budget Estimator Agent


# ============================================================================
# LLM CONFIGURATION
# ============================================================================

# Get API key from environment or Streamlit secrets
def get_api_key():
    """Get Groq API key from environment variables or Streamlit secrets"""
    # Try to get from environment first (local development)
    api_key = os.getenv("GROQ_API_KEY")
    
    # If not found, try Streamlit secrets (cloud deployment)
    if not api_key:
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
                api_key = st.secrets["GROQ_API_KEY"]
        except:
            pass
    
    return api_key

# Shared LLM instance
# Using Groq's free tier with openai/gpt-oss-120b model
api_key = get_api_key()
llm = None

if api_key:
    try:
        llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7, api_key=api_key)
    except Exception as e:
        print(f"Error initializing ChatGroq: {e}")
else:
    print("Warning: GROQ_API_KEY not found. LLM features will be disabled.")



# ============================================================================
# AGENT 1: PLANNER AGENT
# ============================================================================

def planner_node(state: TravelState) -> dict:
    """
    Planner Agent: Extracts structured travel information from user input.
    
    Role: Parse free-text travel requests and extract:
        - Destination
        - Travel dates
        - Budget
        - Preferences (activities, interests, etc.)
    
    Input: user_input from TravelState
    Output: Updates destination, travel_dates, budget, preferences in TravelState
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a travel planning assistant. Extract structured travel intent from the user's request.\n"
         "Return ONLY a valid JSON object with these exact keys: destination, travel_dates, budget, preferences.\n"
         "Do not include any other text, explanations, or markdown formatting.\n"
         "If you cannot determine a value, use an empty string. If destination is unclear, use \"unknown\".\n\n"
         "Example output:\n"
         '{{"destination": "Paris", "travel_dates": "June 15-22", "budget": "$2000", "preferences": "art museums and cafes"}}'),
        ("human", "Travel request: {user_input}"),
    ])
    chain = prompt | llm
    response = chain.invoke({"user_input": state["user_input"]})
    
    try:
        # Clean up response - remove markdown code blocks if present
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        
        parsed = json.loads(content)
        return {
            "destination": parsed.get("destination", "unknown"),
            "travel_dates": parsed.get("travel_dates", ""),
            "budget": parsed.get("budget", ""),
            "preferences": parsed.get("preferences", ""),
        }
    except Exception as e:
        print(f"Warning: Failed to parse planner response: {e}")
        print(f"Raw response: {response.content}")
        return {"destination": "unknown", "travel_dates": "", "budget": "", "preferences": ""}


# ============================================================================
# AGENT 2: RESEARCH AGENT
# ============================================================================

def research_node(state: TravelState) -> dict:
    """
    Research Agent: Gathers destination information and travel tips.
    
    Role: Research the destination and provide:
        - Top attractions and highlights
        - Practical travel tips
        - Local insights based on user preferences
    
    Input: destination, preferences from TravelState
    Output: Updates research_notes in TravelState
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a destination research specialist. Provide at least 3 highlights, top attractions,\n"
         "and practical travel tips for the given destination and traveller preferences.\n"
         "If destination is \"unknown\", provide general travel tips instead."),
        ("human", "Destination: {destination}\nPreferences: {preferences}"),
    ])
    chain = prompt | llm
    response = chain.invoke({
        "destination": state["destination"],
        "preferences": state["preferences"],
    })
    return {"research_notes": response.content}


# ============================================================================
# AGENT 3: ITINERARY BUILDER AGENT
# ============================================================================

def itinerary_node(state: TravelState) -> dict:
    """
    Itinerary Builder Agent: Creates detailed day-by-day travel plans.
    
    Role: Build a structured itinerary that:
        - Organizes activities by day
        - Incorporates research findings
        - Aligns with user preferences and travel dates
    
    Input: destination, travel_dates, preferences, research_notes from TravelState
    Output: Updates itinerary in TravelState
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert travel itinerary builder. Create a detailed day-by-day itinerary.\n"
         "Each day must include at least one activity drawn directly from the research notes provided.\n"
         "Format clearly with Day 1, Day 2, etc."),
        ("human",
         "Destination: {destination}\n"
         "Dates: {travel_dates}\n"
         "Preferences: {preferences}\n"
         "Research Notes: {research_notes}"),
    ])
    chain = prompt | llm
    response = chain.invoke({
        "destination": state["destination"],
        "travel_dates": state["travel_dates"],
        "preferences": state["preferences"],
        "research_notes": state["research_notes"],
    })
    return {"itinerary": response.content}


# ============================================================================
# AGENT 4: BUDGET ESTIMATOR AGENT
# ============================================================================

def budget_node(state: TravelState) -> dict:
    """
    Budget Estimator Agent: Provides cost analysis and budget breakdown.
    
    Role: Analyze the itinerary and provide:
        - Cost breakdown (accommodation, transport, food, activities)
        - Budget feasibility assessment
        - Recommendations for staying within budget
    
    Input: destination, travel_dates, budget, itinerary from TravelState
    Output: Updates budget_estimate in TravelState
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a travel budget analyst. Provide a cost breakdown covering accommodation,\n"
         "transport, food, and activities. If a numeric budget is provided, explicitly state\n"
         "whether the estimated total is within, at, or over that budget."),
        ("human",
         "Destination: {destination}\n"
         "Dates: {travel_dates}\n"
         "Budget: {budget}\n"
         "Itinerary: {itinerary}"),
    ])
    chain = prompt | llm
    response = chain.invoke({
        "destination": state["destination"],
        "travel_dates": state["travel_dates"],
        "budget": state["budget"],
        "itinerary": state["itinerary"],
    })
    return {"budget_estimate": response.content}


# ============================================================================
# LANGGRAPH WORKFLOW DEFINITION
# ============================================================================

def build_graph():
    """
    Build the LangGraph workflow connecting all agents.
    
    Workflow:
        START → Planner → Research → Itinerary Builder → Budget Estimator → END
    
    Each agent processes the shared TravelState and passes it to the next agent.
    """
    graph = StateGraph(TravelState)
    
    # Add nodes (agents)
    graph.add_node("planner", planner_node)
    graph.add_node("research", research_node)
    graph.add_node("itinerary", itinerary_node)
    graph.add_node("budget", budget_node)
    
    # Define workflow edges
    graph.set_entry_point("planner")
    graph.add_edge("planner", "research")
    graph.add_edge("research", "itinerary")
    graph.add_edge("itinerary", "budget")
    graph.add_edge("budget", END)
    
    return graph.compile()


# ============================================================================
# EXECUTION FUNCTION
# ============================================================================

def execute_travel_planning(user_input: str) -> TravelState:
    """
    Execute the multi-agent travel planning workflow.
    
    Args:
        user_input: User's travel request in natural language
    
    Returns:
        TravelState: Final state with all fields populated by agents
    """
    initial_state: TravelState = {
        "user_input": user_input,
        "destination": "",
        "travel_dates": "",
        "budget": "",
        "preferences": "",
        "research_notes": "",
        "itinerary": "",
        "budget_estimate": "",
    }
    
    graph = build_graph()
    final_state = graph.invoke(initial_state)
    return final_state


# ============================================================================
# CLI MODE
# ============================================================================

def print_travel_plan(state: TravelState) -> None:
    """Print travel plan to console (CLI mode)"""
    print("\n" + "="*80)
    print("TRAVEL PLAN GENERATED")
    print("="*80)
    
    print("\n📍 DESTINATION")
    print("-" * 80)
    print(state["destination"] if state["destination"] else "Not determined")
    
    print("\n📅 TRAVEL DATES")
    print("-" * 80)
    print(state["travel_dates"] if state["travel_dates"] else "Not specified")
    
    print("\n💰 BUDGET")
    print("-" * 80)
    print(state["budget"] if state["budget"] else "Not specified")
    
    print("\n🎯 PREFERENCES")
    print("-" * 80)
    print(state["preferences"] if state["preferences"] else "Not specified")
    
    print("\n📝 ITINERARY")
    print("-" * 80)
    print(state["itinerary"] if state["itinerary"] else "No itinerary generated")
    
    print("\n💵 BUDGET ESTIMATE")
    print("-" * 80)
    print(state["budget_estimate"] if state["budget_estimate"] else "No estimate generated")
    
    print("\n" + "="*80)


def main() -> None:
    """Main function for CLI mode"""
    print("="*80)
    print("MULTI-AGENT TRAVEL PLANNER")
    print("="*80)
    print("\nThis system uses 4 AI agents to create your perfect travel plan:")
    print("  1. Planner Agent - Extracts travel details")
    print("  2. Research Agent - Finds attractions and tips")
    print("  3. Itinerary Builder - Creates day-by-day plans")
    print("  4. Budget Estimator - Analyzes costs")
    print("\n" + "="*80 + "\n")
    
    # Get user input
    user_input = ""
    while not user_input.strip():
        user_input = input("Enter your travel request: ")
    
    print("\n🔄 Processing your request through 4 specialized agents...")
    print("   This may take 15-30 seconds...\n")
    
    # Execute workflow
    final_state = execute_travel_planning(user_input)
    
    # Display results
    print_travel_plan(final_state)


# ============================================================================
# STREAMLIT MODE
# ============================================================================

def run_streamlit_app():
    """Run Streamlit web interface"""
    import streamlit as st
    
    # Page configuration
    st.set_page_config(
        page_title="Multi-Agent Travel Planner",
        page_icon="✈️",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: #1E88E5;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.2rem;
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
        }
        .agent-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .result-section {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1E88E5;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">✈️ Multi-Agent Travel Planner</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Powered by LangChain + LangGraph | 4 Specialized AI Agents</div>', unsafe_allow_html=True)
    
    # Sidebar - Agent Information
    with st.sidebar:
        st.header("🤖 Agent System")
        st.markdown("---")
        
        st.markdown("### Agent 1: Planner")
        st.markdown("📋 Extracts destination, dates, budget, and preferences from your request")
        
        st.markdown("### Agent 2: Research")
        st.markdown("🔍 Gathers attractions, highlights, and travel tips")
        
        st.markdown("### Agent 3: Itinerary Builder")
        st.markdown("📅 Creates detailed day-by-day travel plans")
        
        st.markdown("### Agent 4: Budget Estimator")
        st.markdown("💰 Provides cost breakdown and budget analysis")
        
        st.markdown("---")
        st.markdown("**Technology Stack:**")
        st.markdown("- LangGraph for workflow")
        st.markdown("- LangChain for LLM integration")
        st.markdown("- Groq API (openai/gpt-oss-120b)")
        st.markdown("- Streamlit for UI")
    
    # Main content
    if not api_key:
        st.error("⚠️ **GROQ_API_KEY not found!** Please set your API key in the `.env` file to enable travel plan generation.")
        st.info("You can get a free API key from [console.groq.com](https://console.groq.com)")
    
    st.markdown("### 📝 Enter Your Travel Request")

    st.markdown("Describe your ideal trip in natural language. Include destination, dates, budget, and preferences.")
    
    # Example requests
    with st.expander("💡 See Example Requests"):
        st.markdown("""
        - "I want to visit Tokyo for 7 days in March with a budget of $3000. I love food and temples."
        - "Plan a romantic trip to Paris for 5 days in June. Budget is $2500. We enjoy art museums and cafes."
        - "Family vacation to Bali for 10 days in December. Budget $5000. Kids love beaches and animals."
        - "Solo backpacking trip to Thailand for 2 weeks. Budget $1500. Interested in culture and nightlife."
        """)
    
    # User input
    user_input = st.text_area(
        "Your Travel Request:",
        height=100,
        placeholder="Example: I want to travel to Tokyo for 7 days in March with a budget of $3000. I love food and temples."
    )
    
    # Generate button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        generate_button = st.button("🚀 Generate Travel Plan", use_container_width=True, type="primary")
    
    # Process request
    if generate_button:
        if not user_input.strip():
            st.error("⚠️ Please enter a travel request")
        else:
            # Progress indicator
            with st.spinner("🔄 Processing through 4 AI agents... This may take 15-30 seconds..."):
                try:
                    # Execute workflow
                    final_state = execute_travel_planning(user_input)
                    
                    # Display success
                    st.success("✅ Travel plan generated successfully!")
                    
                    # Display results in tabs
                    tab1, tab2, tab3, tab4 = st.tabs(["📍 Overview", "📝 Itinerary", "💵 Budget", "🔍 Research"])
                    
                    with tab1:
                        st.markdown("### 📍 Trip Overview")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Destination:**")
                            st.info(final_state["destination"] or "Not determined")
                            
                            st.markdown("**Budget:**")
                            st.info(final_state["budget"] or "Not specified")
                        
                        with col2:
                            st.markdown("**Travel Dates:**")
                            st.info(final_state["travel_dates"] or "Not specified")
                            
                            st.markdown("**Preferences:**")
                            st.info(final_state["preferences"] or "Not specified")
                    
                    with tab2:
                        st.markdown("### 📝 Day-by-Day Itinerary")
                        if final_state["itinerary"]:
                            st.markdown(final_state["itinerary"])
                        else:
                            st.warning("No itinerary generated")
                    
                    with tab3:
                        st.markdown("### 💵 Budget Breakdown")
                        if final_state["budget_estimate"]:
                            st.markdown(final_state["budget_estimate"])
                        else:
                            st.warning("No budget estimate generated")
                    
                    with tab4:
                        st.markdown("### 🔍 Destination Research")
                        if final_state["research_notes"]:
                            st.markdown(final_state["research_notes"])
                        else:
                            st.warning("No research notes generated")
                    
                    # Download option
                    st.markdown("---")
                    travel_plan_text = f"""
TRAVEL PLAN
===========

Destination: {final_state['destination']}
Travel Dates: {final_state['travel_dates']}
Budget: {final_state['budget']}
Preferences: {final_state['preferences']}

ITINERARY
=========
{final_state['itinerary']}

BUDGET ESTIMATE
===============
{final_state['budget_estimate']}

RESEARCH NOTES
==============
{final_state['research_notes']}
"""
                    st.download_button(
                        label="📥 Download Travel Plan",
                        data=travel_plan_text,
                        file_name="travel_plan.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.error("Please check your GROQ_API_KEY in .env file")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Check if running in Streamlit
    try:
        import streamlit as st
        # If streamlit is imported and we're in streamlit context
        if hasattr(st, 'runtime') and st.runtime.exists():
            run_streamlit_app()
        else:
            # Streamlit installed but not running in streamlit context
            main()
    except ImportError:
        # Streamlit not installed, run CLI mode
        main()
