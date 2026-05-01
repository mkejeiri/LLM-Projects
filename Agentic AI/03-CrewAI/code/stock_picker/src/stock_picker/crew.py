# crew.py - Orchestration module for the Stock Picker crew
# Demonstrates: structured outputs (output_pydantic), hierarchical process,
# custom tool (PushNotificationTool), manager agent with delegation,
# callbacks (@before_kickoff, @after_kickoff, task_callback),
# and task guardrails for output validation

from crewai import Agent, Crew, Process, Task, TaskOutput
from crewai.project import CrewBase, agent, crew, task, before_kickoff, after_kickoff
from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field
from typing import List, Tuple, Any
from datetime import datetime
from .tools.push_tool import PushNotificationTool


# --- Structured Output Schemas ---
# These Pydantic models constrain task outputs to specific JSON formats
# This guides agents to produce exactly the information we need

class TrendingCompany(BaseModel):
    """A company that is in the news and attracting attention"""
    name: str = Field(description="Company name")
    ticker: str = Field(description="Stock ticker symbol")
    reason: str = Field(description="Reason this company is trending in the news")


class TrendingCompanyList(BaseModel):
    """List of multiple trending companies that are in the news"""
    companies: List[TrendingCompany] = Field(description="List of companies trending in the news")


class TrendingCompanyResearch(BaseModel):
    """Detailed research on a company"""
    name: str = Field(description="Company name")
    market_position: str = Field(description="Current market position and competitive analysis")
    future_outlook: str = Field(description="Future outlook and growth prospects")
    investment_potential: str = Field(description="Investment potential and suitability for investment")


class TrendingCompanyResearchList(BaseModel):
    """A list of detailed research on all the companies"""
    research_list: List[TrendingCompanyResearch] = Field(description="Comprehensive research on all trending companies")


# --- Guardrail Function ---
# Guardrails validate task output BEFORE it's passed to the next task.
# Returns (True, result) if valid, or (False, "error message") if invalid.
# On failure, the error is sent back to the agent to retry (up to guardrail_max_retries).

def validate_pick_has_rationale(result: TaskOutput) -> Tuple[bool, Any]:
    """Ensure the final stock pick includes a clear rationale (at least 3 sentences)."""
    sentences = [s.strip() for s in result.raw.split('.') if len(s.strip()) > 10]
    if len(sentences) < 3:
        # Returning (False, error_msg) sends this feedback to the agent for retry
        return (False, "Your stock pick must include a rationale of at least 3 sentences explaining why.")
    return (True, result.raw)


# --- Crew-Level Task Callback ---
# This function fires after EVERY task completes. Useful for monitoring/logging.

def log_task_completion(output: TaskOutput):
    """Called after each task finishes — for progress monitoring."""
    print(f"  ✓ Task completed: {output.description[:60]}...")
    print(f"    Output length: {len(output.raw)} chars")


# --- Crew Definition ---

@CrewBase
class StockPicker():
    """StockPicker crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # --- @before_kickoff ---
    # Runs BEFORE the crew starts. Receives the inputs dict, can modify it.
    # Use case: enrich inputs with dynamic data (timestamps, market state, etc.)
    # Must return the modified inputs dict.
    @before_kickoff
    def prepare_inputs(self, inputs):
        """Add current date to inputs so agents know the timeframe."""
        inputs['current_date'] = str(datetime.now().date())
        print(f"[before_kickoff] Starting stock picker for sector: {inputs.get('sector')}")
        return inputs

    # --- @after_kickoff ---
    # Runs AFTER the crew finishes. Receives the CrewOutput, can modify it.
    # Use case: post-processing, logging, saving results to DB.
    # Must return the modified output.
    @after_kickoff
    def process_output(self, output):
        """Log the final decision after crew completes."""
        print(f"[after_kickoff] Crew finished. Decision length: {len(output.raw)} chars")
        return output

    # Trending company finder uses SerperDevTool to search news
    @agent
    def trending_company_finder(self) -> Agent:
        return Agent(
            config=self.agents_config['trending_company_finder'],
            tools=[SerperDevTool()]
        )

    # Financial researcher also uses SerperDevTool for deep analysis
    @agent
    def financial_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['financial_researcher'],
            tools=[SerperDevTool()]
        )

    # Stock picker gets the custom PushNotificationTool to notify the user
    @agent
    def stock_picker(self) -> Agent:
        return Agent(
            config=self.agents_config['stock_picker'],
            tools=[PushNotificationTool()]
        )

    # Task with output_pydantic: forces output to conform to TrendingCompanyList schema
    @task
    def find_trending_companies(self) -> Task:
        return Task(
            config=self.tasks_config['find_trending_companies'],
            output_pydantic=TrendingCompanyList,
        )

    # Research task also uses structured output for consistent data format
    @task
    def research_trending_companies(self) -> Task:
        return Task(
            config=self.tasks_config['research_trending_companies'],
            output_pydantic=TrendingCompanyResearchList,
        )

    # Final task with GUARDRAIL: validates the output before accepting it.
    # If validate_pick_has_rationale returns (False, error), the agent retries.
    @task
    def pick_best_company(self) -> Task:
        return Task(
            config=self.tasks_config['pick_best_company'],
            guardrail=validate_pick_has_rationale,  # Validate output quality
            guardrail_max_retries=3,                # Agent gets 3 attempts to fix
        )

    @crew
    def crew(self) -> Crew:
        """Creates the StockPicker crew"""

        # Manager agent is created separately - NOT in the @agent decorated list
        manager = Agent(
            config=self.agents_config['manager'],
            allow_delegation=True
        )

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            verbose=True,
            manager_agent=manager,
            task_callback=log_task_completion,  # Fires after each task for monitoring
        )
