# crew.py - The orchestration module that brings agents, tasks, and crew together
# Uses CrewAI decorators to auto-wire YAML configs to Python objects

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task


# @CrewBase enables YAML config loading and auto-collection of agents/tasks
@CrewBase
class Debate():
    """Debate crew"""

    # Path to YAML config files (relative to this module)
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # @agent decorator registers this Agent into self.agents automatically
    # Method name 'debater' must match the key in agents.yaml
    @agent
    def debater(self) -> Agent:
        return Agent(
            config=self.agents_config['debater'],
            verbose=True
        )

    # Second agent - the judge uses a different LLM (Claude) defined in YAML
    @agent
    def judge(self) -> Agent:
        return Agent(
            config=self.agents_config['judge'],
            verbose=True
        )

    # @task decorator registers this Task into self.tasks automatically
    # Method name 'propose' must match the key in tasks.yaml
    @task
    def propose(self) -> Task:
        return Task(config=self.tasks_config['propose'])

    # Same debater agent handles this task - task description drives behavior
    @task
    def oppose(self) -> Task:
        return Task(config=self.tasks_config['oppose'])

    # Judge evaluates both arguments in sequential order
    @task
    def decide(self) -> Task:
        return Task(config=self.tasks_config['decide'])

    # @crew decorator marks the method that assembles the final Crew object
    @crew
    def crew(self) -> Crew:
        """Creates the Debate crew"""
        return Crew(
            agents=self.agents,   # Auto-populated by @agent decorators
            tasks=self.tasks,     # Auto-populated by @task decorators
            process=Process.sequential,  # Tasks run in order: propose -> oppose -> decide
            verbose=True,
        )
