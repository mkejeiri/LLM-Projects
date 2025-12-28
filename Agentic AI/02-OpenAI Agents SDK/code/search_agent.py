# SEARCH AGENT - Web Research Execution Component
# Demonstrates tool integration and cost optimization in agent design
# Key concept: Specialized agent with web search capabilities and concise summarization

from agents import Agent, WebSearchTool, ModelSettings

# Concise summarization instructions optimized for synthesis workflows
INSTRUCTIONS = (
    "You are a research assistant. Given a search term, you search the web for that term and "
    "produce a concise summary of the results. The summary must 2-3 paragraphs and less than 300 "
    "words. Capture the main points. Write succintly, no need to have complete sentences or good "
    "grammar. This will be consumed by someone synthesizing a report, so its vital you capture the "
    "essence and ignore any fluff. Do not include any additional commentary other than the summary itself."
)

# Agent with tool integration and cost controls
# search_context_size="low" reduces API costs while maintaining research quality
search_agent = Agent(
    name="Search agent",
    instructions=INSTRUCTIONS,
    tools=[WebSearchTool(search_context_size="low")],  # Cost optimization parameter
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="required"),  # Ensures tool usage
)