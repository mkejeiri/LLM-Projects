#!/usr/bin/env python
# main.py - Entry point that sets the sector and kicks off the stock picker crew
#
# Note: @before_kickoff in crew.py will enrich these inputs with current_date
# before the crew starts. @after_kickoff will log results after completion.

import os
from stock_picker.crew import StockPicker

# Create output directory for JSON results and final decision
os.makedirs('output', exist_ok=True)


def run():
    """Run the crew."""
    # {sector} template variable gets interpolated into YAML configs
    # @before_kickoff will add 'current_date' to this dict automatically
    inputs = {
        'sector': 'Technology',
    }

    # Kicks off hierarchical process: manager delegates tasks to agents
    # Execution order: @before_kickoff -> tasks (with task_callback) -> @after_kickoff
    result = StockPicker().crew().kickoff(inputs=inputs)

    print("\n\n=== FINAL DECISION ===\n\n")
    print(result.raw)
