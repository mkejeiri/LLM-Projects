#!/usr/bin/env python
# main.py - Entry point that defines template variable values and kicks off the crew

import warnings
import os

from debate.crew import Debate

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Create output directory for task results
os.makedirs('output', exist_ok=True)


def run():
    """Run the crew."""
    # inputs dict provides values for {motion} template variables in YAML configs
    inputs = {
        'motion': 'There needs to be strict laws to regulate LLMs'
    }
    # Instantiate class -> call crew() method -> kickoff with inputs
    result = Debate().crew().kickoff(inputs=inputs)
    # result.raw contains the final task's text output (judge's decision)
    print(result.raw)
