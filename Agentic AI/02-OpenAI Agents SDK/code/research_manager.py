# RESEARCH MANAGER - Multi-Agent Orchestration and Workflow Control
# Demonstrates async coordination, generator patterns, and trace management
# Key concepts: Agent handoffs, parallel execution, and streaming updates

from agents import Runner, trace, gen_trace_id
from search_agent import search_agent
from planner_agent import planner_agent, WebSearchItem, WebSearchPlan
from writer_agent import writer_agent, ReportData
from email_agent import email_agent
import asyncio

class ResearchManager:
    """Orchestrates the multi-agent research workflow with streaming updates"""

    async def run(self, query: str):
        """Main workflow - demonstrates generator pattern for streaming UI updates"""
        trace_id = gen_trace_id()
        with trace("Research trace", trace_id=trace_id):  # OpenAI trace management
            print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}")
            # yield creates an async generator that streams progress updates to the UI
            # This allows real-time feedback instead of waiting for the entire process to complete
            yield f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}"
            print("Starting research...")
            
            # Sequential agent handoffs with status updates
            search_plan = await self.plan_searches(query)
            yield "Searches planned, starting to search..."     
            search_results = await self.perform_searches(search_plan)
            yield "Searches complete, writing report..."
            report = await self.write_report(query, search_results)
            yield "Report written, sending email..."
            await self.send_email(report)
            yield "Email sent, research complete"
            yield report.markdown_report  # Final deliverable

    async def plan_searches(self, query: str) -> WebSearchPlan:
        """Agent handoff: Query → Search Plan"""
        print("Planning searches...")
        result = await Runner.run(planner_agent, f"Query: {query}")
        print(f"Will perform {len(result.final_output.searches)} searches")
        return result.final_output_as(WebSearchPlan)

    async def perform_searches(self, search_plan: WebSearchPlan) -> list[str]:
        """Parallel execution of search agents with progress tracking"""
        print("Searching...")
        num_completed = 0
        # Async parallel execution - key performance optimization
        tasks = [asyncio.create_task(self.search(item)) for item in search_plan.searches]
        results = []
        
        # Process completions as they arrive (not in order)
        for task in asyncio.as_completed(tasks):
            result = await task
            if result is not None:
                results.append(result)
            num_completed += 1
            print(f"Searching... {num_completed}/{len(tasks)} completed")
        print("Finished searching")
        return results

    async def search(self, item: WebSearchItem) -> str | None:
        """Individual search execution with error handling"""
        input = f"Search term: {item.query}\nReason for searching: {item.reason}"
        try:
            result = await Runner.run(search_agent, input)
            return str(result.final_output)
        except Exception:
            return None  # Graceful degradation on search failures

    async def write_report(self, query: str, search_results: list[str]) -> ReportData:
        """Synthesis phase: Raw data → Structured report"""
        print("Thinking about report...")
        input = f"Original query: {query}\nSummarized search results: {search_results}"
        result = await Runner.run(writer_agent, input)
        print("Finished writing report")
        return result.final_output_as(ReportData)
    
    async def send_email(self, report: ReportData) -> None:
        """Final delivery phase: Report → External communication"""
        print("Writing email...")
        result = await Runner.run(email_agent, report.markdown_report)
        print("Email sent")
        return report