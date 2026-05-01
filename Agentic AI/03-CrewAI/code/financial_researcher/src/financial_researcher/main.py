#!/usr/bin/env python
# main.py - Entry point that sets the company to research and kicks off the crew

import os
from financial_researcher.crew import ResearchCrew

# Create output directory for the final report
os.makedirs('output', exist_ok=True)


def run():
    """Run the research crew."""
    # {company} template variable gets interpolated into all YAML configs
    inputs = {
        'company': 'Tesla'
    }

    # Kick off the sequential pipeline: research_task -> analysis_task
    result = ResearchCrew().crew().kickoff(inputs=inputs)

    print("\n\n=== FINAL REPORT ===\n\n")
    print(result.raw)
    print("\n\nReport has been saved to output/report.md")
