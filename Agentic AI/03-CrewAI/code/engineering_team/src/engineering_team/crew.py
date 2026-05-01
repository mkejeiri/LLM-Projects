# crew.py - Orchestration module for the Engineering Team crew
# Demonstrates: multi-agent collaboration, task context/dependencies, selective code execution
#
# Key design decisions from the transcript:
# - Engineering lead does NOT get code execution (only designs)
# - Frontend engineer does NOT get code execution (running Gradio in Docker = different challenge)
# - Backend + test engineers DO get code execution (they need to run/verify code)
# - Think of tasks as user prompts and agents as system prompts

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class EngineeringTeam():
    """EngineeringTeam crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # Engineering lead uses GPT-4o for high-level design/planning
    # No code execution - it only produces a design document
    @agent
    def engineering_lead(self) -> Agent:
        return Agent(
            config=self.agents_config['engineering_lead'],
            verbose=True,
        )

    # Backend engineer uses Claude and can execute code in Docker
    # Gets generous timeout (500s) because implementation can take a few minutes
    @agent
    def backend_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['backend_engineer'],
            verbose=True,
            allow_code_execution=True,
            code_execution_mode="safe",
            max_execution_time=500,
            max_retry_limit=3
        )

    # Frontend engineer builds a Gradio UI but does NOT execute it
    # Running Gradio in Docker would start a web server inside the container
    @agent
    def frontend_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['frontend_engineer'],
            verbose=True,
        )

    # Test engineer writes AND runs unit tests in Docker to verify the backend
    @agent
    def test_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['test_engineer'],
            verbose=True,
            allow_code_execution=True,
            code_execution_mode="safe",
            max_execution_time=500,
            max_retry_limit=3
        )

    @task
    def design_task(self) -> Task:
        return Task(config=self.tasks_config['design_task'])

    @task
    def code_task(self) -> Task:
        return Task(config=self.tasks_config['code_task'])

    @task
    def frontend_task(self) -> Task:
        return Task(config=self.tasks_config['frontend_task'])

    @task
    def test_task(self) -> Task:
        return Task(config=self.tasks_config['test_task'])

    # The crew function only needs agents, tasks, process, verbose
    # Don't let autocomplete add unnecessary complexity ("vibe coding" warning)
    @crew
    def crew(self) -> Crew:
        """Creates the Engineering Team crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
