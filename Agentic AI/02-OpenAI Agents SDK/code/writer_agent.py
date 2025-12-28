# WRITER AGENT - Report Synthesis and Structured Output Component
# Demonstrates advanced instruction engineering and multi-faceted data structures
# Key concepts: Two-phase generation, executive summaries, and research continuity

from pydantic import BaseModel, Field
from agents import Agent

# Two-phase instruction design: outline → report generation
# Mirrors human research methodology for better organization
INSTRUCTIONS = (
    "You are a senior researcher tasked with writing a cohesive report for a research query. "
    "You will be provided with the original query, and some initial research done by a research assistant.\n"
    "You should first come up with an outline for the report that describes the structure and "
    "flow of the report. Then, generate the report and return that as your final output.\n"
    "The final output should be in markdown format, and it should be lengthy and detailed. Aim "
    "for 5-10 pages of content, at least 1000 words."
)

# Multi-faceted output schema for comprehensive reporting
class ReportData(BaseModel):
    # Executive summary for quick decision-making
    short_summary: str = Field(description="A short 2-3 sentence summary of the findings.")
    # Primary deliverable - comprehensive analysis
    markdown_report: str = Field(description="The final report")
    # Research continuity - enables iterative investigation
    follow_up_questions: list[str] = Field(description="Suggested topics to research further")

# Agent optimized for synthesis and structured output generation
writer_agent = Agent(
    name="WriterAgent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=ReportData,  # Enforces structured output compliance
)