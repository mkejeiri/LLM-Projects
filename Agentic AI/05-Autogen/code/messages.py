# messages.py — Shared message type and utility for finding peer agents.
# Kept in a separate module to minimize code in the agent template (fewer tokens for the LLM).
# find_recipient() discovers other spawned agents by scanning the directory for agentN.py files.

from dataclasses import dataclass
from autogen_core import AgentId
import glob
import os
import random


@dataclass
class Message:
    content: str


def find_recipient() -> AgentId:
    """Find a random peer agent to bounce ideas off of.
    Scans the directory for agent*.py files (agent1.py, agent2.py, etc.)
    and returns a randomly selected AgentId. May return self — that's OK."""
    try:
        agent_files = glob.glob("agent*.py")
        agent_names = [os.path.splitext(file)[0] for file in agent_files]
        agent_names.remove("agent")  # Remove the template itself
        agent_name = random.choice(agent_names)
        print(f"Selecting agent for refinement: {agent_name}")
        return AgentId(agent_name, "default")
    except Exception as e:
        print(f"Exception finding recipient: {e}")
        return AgentId("agent1", "default")
