#!/usr/bin/env python
# main.py - Entry point that sets the coding assignment and kicks off the coder crew

import warnings
import os

from coder.crew import Coder

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Create output directory for code results
os.makedirs('output', exist_ok=True)

# The assignment is deliberately complex to prove the agent actually runs the code
# (not just predicting the output from training data)
# This series 1 - 1/3 + 1/5 - 1/7 + ... multiplied by 4 approximates pi
# With 10,000 terms it will be ~3.14149 (not exact pi) proving real execution
assignment = 'Write a python program to calculate the first 10,000 terms \
    of this series, multiplying the total by 4: 1 - 1/3 + 1/5 - 1/7 + ...'


def run():
    """Run the crew."""
    inputs = {
        'assignment': assignment,
    }
    result = Coder().crew().kickoff(inputs=inputs)
    print(result.raw)
