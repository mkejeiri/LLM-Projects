#!/usr/bin/env python
# main.py - Entry point that defines requirements and kicks off the engineering team crew
#
# Why this assignment: The generated trading account system gets reused in week 6
# of the course for building agent traders. This is a "two for one" - building the
# crew AND generating useful code for future projects.

import warnings
import os

from engineering_team.crew import EngineeringTeam

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Create output directory for generated code and design docs
os.makedirs('output', exist_ok=True)

# The requirements that the engineering team will implement
# Deliberately complex to test the crew's ability to coordinate across agents
requirements = """
A simple account management system for a trading simulation platform.
The system should allow users to create an account, deposit funds, and withdraw funds.
The system should allow users to record that they have bought or sold shares, providing a quantity.
The system should calculate the total value of the user's portfolio, and the profit or loss from the initial deposit.
The system should be able to report the holdings of the user at any point in time.
The system should be able to report the profit or loss of the user at any point in time.
The system should be able to list the transactions that the user has made over time.
The system should prevent the user from withdrawing funds that would leave them with a negative balance, or
 from buying more shares than they can afford, or selling shares that they don't have.
The system has access to a function get_share_price(symbol) which returns the current price of a share,
 and includes a test implementation that returns fixed prices for AAPL, TSLA, GOOGL.
"""

# Template variables interpolated into YAML configs (even in output_file paths)
module_name = "accounts.py"
class_name = "Account"


def run():
    """Run the crew."""
    inputs = {
        'requirements': requirements,
        'module_name': module_name,
        'class_name': class_name
    }
    # Kicks off sequential pipeline: design -> code -> frontend -> test (~5 min total)
    result = EngineeringTeam().crew().kickoff(inputs=inputs)
    print(result.raw)
