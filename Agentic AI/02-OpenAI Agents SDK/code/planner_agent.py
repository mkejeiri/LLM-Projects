# PLANNER AGENT - Multi-Agent Research System Entry Point
# Demonstrates query decomposition and search strategy planning
# Key concept: Breaking complex queries into targeted web searches

from pydantic import BaseModel, Field
from agents import Agent

# Search quantity configuration - balance between thoroughness and cost
HOW_MANY_SEARCHES = 5

# Strategic planning instructions for search decomposition
INSTRUCTIONS = f"You are a helpful research assistant. Given a query, come up with a set of web searches \
to perform to best answer the query. Output {HOW_MANY_SEARCHES} terms to query for."

# Individual search item with reasoning - enables explainable AI
class WebSearchItem(BaseModel):
    reason: str = Field(description="Your reasoning for why this search is important to the query.")
    query: str = Field(description="The search term to use for the web search.")

# Search plan container - structured output for downstream agents
class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem] = Field(description="A list of web searches to perform to best answer the query.")
    
# Agent configuration optimized for planning tasks
planner_agent = Agent(
    name="PlannerAgent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=WebSearchPlan,
)