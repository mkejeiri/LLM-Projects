# crew.py - Orchestration module for the Financial Researcher crew
# Demonstrates: SerperDevTool for web search, task context for passing data between agents

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool  # Google search via Serper API (requires SERPER_API_KEY)


@CrewBase
class ResearchCrew():
    """Research crew for comprehensive topic analysis and reporting"""

    # Researcher agent gets SerperDevTool to search the web for current information
    # Without this tool, the agent relies only on LLM training data (stale)
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True,
            tools=[SerperDevTool()]  # Gives the agent ability to Google search
        )

    # Analyst agent has no tools - it works purely from the context passed to it
    @agent
    def analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['analyst'],
            verbose=True
        )

    # Research task: agent uses SerperDevTool to find current info about the company
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task']
        )

    # Analysis task: receives research_task output via 'context' field in YAML
    # output_file writes the final report to disk
    @task
    def analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['analysis_task'],
            output_file='output/report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the research crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,  # research_task runs first, then analysis_task
            verbose=True,
        )
