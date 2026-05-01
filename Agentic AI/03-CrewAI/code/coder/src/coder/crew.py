# crew.py - Orchestration module for the Coder crew
# Demonstrates: code execution in Docker containers (the "coder agent" pattern)
# Requires: Docker Desktop installed and running

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# One-click install for Docker Desktop:
# https://docs.docker.com/desktop/


@CrewBase
class Coder():
    """Coder crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def coder(self) -> Agent:
        return Agent(
            config=self.agents_config['coder'],
            verbose=True,
            # allow_code_execution=True gives the agent the ability to write AND run Python
            allow_code_execution=True,
            # "safe" mode runs code inside a Docker container (sandboxed, no access to host)
            # Without Docker, you can omit this but code runs directly on your machine
            code_execution_mode="safe",
            # Timeout in seconds - prevents infinite loops
            max_execution_time=30,
            # Number of retries if code execution fails (syntax errors, runtime errors)
            max_retry_limit=3
        )

    @task
    def coding_task(self) -> Task:
        return Task(
            config=self.tasks_config['coding_task'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Coder crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
