"""
sidekick.py — The Sidekick Agent Class

This module contains the core Sidekick class which encapsulates:
- State definition (what information flows through the graph)
- Structured output schema (how the evaluator responds)
- Worker node (the assistant that does the actual work)
- Evaluator node (assesses the worker's output)
- Graph building (wiring nodes and edges together)
- Super step execution (invoking the graph)

The Sidekick implements the Evaluator-Optimizer pattern:
Worker does work → Evaluator assesses → Accept OR send back for retry
"""

import os
from typing import Annotated, List, Any, Optional, Dict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sidekick_tools import playwright_tools, other_tools
import uuid
import asyncio
from datetime import datetime

load_dotenv(override=True)


# --- STATE DEFINITION ---
# This TypedDict defines ALL information that flows through the graph.
# Only 'messages' has a reducer (add_messages) — it accumulates.
# All other fields are simple values that get overwritten by the latest node.
class State(TypedDict):
    messages: Annotated[List[Any], add_messages]  # Conversation history (accumulates via reducer)
    success_criteria: str                          # What the user considers "done" (set once)
    feedback_on_work: Optional[str]               # Evaluator's feedback to worker (set by evaluator)
    success_criteria_met: bool                     # Has the task succeeded? (set by evaluator)
    user_input_needed: bool                        # Does the user need to intervene? (set by evaluator)


# --- STRUCTURED OUTPUT SCHEMA ---
# The evaluator LLM MUST respond with JSON conforming to this schema.
# LangChain's with_structured_output() enforces this automatically.
class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback on the assistant's response")
    success_criteria_met: bool = Field(description="Whether the success criteria have been met")
    user_input_needed: bool = Field(
        description="True if more input is needed from the user, or clarifications, or the assistant is stuck"
    )


class Sidekick:
    """
    The Sidekick agent — a multi-agent system with worker + evaluator.

    Lifecycle:
    1. __init__() — sync initialization (instance variables)
    2. setup() — async initialization (tools, LLMs, graph building)
    3. run_superstep() — execute one user interaction through the graph
    4. cleanup() — close browser/playwright resources
    """

    def __init__(self):
        """Sync init — just set up instance variables. No async work here."""
        self.worker_llm_with_tools = None
        self.evaluator_llm_with_output = None
        self.tools = None
        self.graph = None
        self.sidekick_id = str(uuid.uuid4())  # Unique thread ID for checkpointing
        self.memory = MemorySaver()            # In-memory checkpointer (swap to SqliteSaver for persistence)
        self.browser = None
        self.playwright = None

    async def setup(self):
        """
        Async initialization — must be called after __init__().

        This is separate because Python's __init__ cannot be async,
        but Playwright and graph building require await.
        """
        # Get Playwright browser tools + refs for cleanup
        self.tools, self.browser, self.playwright = await playwright_tools()
        # Add all other tools (search, files, wiki, python, push)
        self.tools += await other_tools()

        # Worker LLM: bound to all tools so it can call any of them
        worker_llm = ChatOpenAI(model="gpt-4o-mini", base_url=os.environ["OPENROUTER_BASE_URL"], api_key=os.environ["OPENROUTER_API_KEY"])
        self.worker_llm_with_tools = worker_llm.bind_tools(self.tools)

        # Evaluator LLM: uses structured output (returns EvaluatorOutput object)
        evaluator_llm = ChatOpenAI(model="gpt-4o-mini", base_url=os.environ["OPENROUTER_BASE_URL"], api_key=os.environ["OPENROUTER_API_KEY"])
        self.evaluator_llm_with_output = evaluator_llm.with_structured_output(EvaluatorOutput)

        # Build the graph (the 5 steps)
        await self.build_graph()

    # --- WORKER NODE ---
    def worker(self, state: State) -> Dict[str, Any]:
        """
        The worker (assistant) node — does the actual work using tools.

        Takes the current state, builds a system prompt with context,
        invokes the LLM with tools, and returns the response.
        """
        # Build system prompt with success criteria and helpful hints
        system_message = f"""You are a helpful assistant that can use tools to complete tasks.
    You keep working on a task until either you have a question or clarification for the user, or the success criteria is met.
    You have many tools to help you, including tools to browse the internet, navigating and retrieving web pages.
    You have a tool to run python code, but note that you would need to include a print() statement if you wanted to receive output.
    The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    This is the success criteria:
    {state["success_criteria"]}
    You should reply either with a question for the user about this assignment, or with your final response.
    If you have a question for the user, you need to reply by clearly stating your question. An example might be:

    Question: please clarify whether you want a summary or a detailed answer

    If you've finished, reply with the final answer, and don't ask a question; simply reply with the answer.
    """
        # If the evaluator previously rejected the work, include that feedback
        if state.get("feedback_on_work"):
            system_message += f"""
    Previously you thought you completed the assignment, but your reply was rejected because the success criteria was not met.
    Here is the feedback on why this was rejected:
    {state["feedback_on_work"]}
    With this feedback, please continue the assignment, ensuring that you meet the success criteria or have a question for the user."""

        # Insert or replace the system message in the conversation
        found_system_message = False
        messages = state["messages"]
        for message in messages:
            if isinstance(message, SystemMessage):
                message.content = system_message
                found_system_message = True

        if not found_system_message:
            messages = [SystemMessage(content=system_message)] + messages

        # Call the LLM — it will either respond or request a tool call
        response = self.worker_llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # --- WORKER ROUTER ---
    def worker_router(self, state: State) -> str:
        """
        Conditional edge: decides where to go after the worker runs.
        - If worker requested a tool call → "tools" (execute the tool)
        - Otherwise → "evaluator" (assess the response)
        """
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        else:
            return "evaluator"

    # --- UTILITY ---
    def format_conversation(self, messages: List[Any]) -> str:
        """Convert message objects to readable 'User: ... / Assistant: ...' format for the evaluator."""
        conversation = "Conversation history:\n\n"
        for message in messages:
            if isinstance(message, HumanMessage):
                conversation += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                text = message.content or "[Tools use]"
                conversation += f"Assistant: {text}\n"
        return conversation

    # --- EVALUATOR NODE ---
    def evaluator(self, state: State) -> State:
        """
        The evaluator node — assesses the worker's response.

        Uses structured output to return a typed decision:
        - feedback (str): what was good/bad
        - success_criteria_met (bool): is the task done?
        - user_input_needed (bool): should we ask the user?
        """
        last_response = state["messages"][-1].content

        system_message = """You are an evaluator that determines if a task has been completed successfully by an Assistant.
    Assess the Assistant's last response based on the given criteria. Respond with your feedback, and with your decision on whether the success criteria has been met,
    and whether more input is needed from the user."""

        user_message = f"""You are evaluating a conversation between the User and Assistant. You decide what action to take based on the last response from the Assistant.

    The entire conversation with the assistant, with the user's original request and all replies, is:
    {self.format_conversation(state["messages"])}

    The success criteria for this assignment is:
    {state["success_criteria"]}

    And the final response from the Assistant that you are evaluating is:
    {last_response}

    Respond with your feedback, and decide if the success criteria is met by this response.
    Also, decide if more user input is required, either because the assistant has a question, needs clarification, or seems to be stuck and unable to answer without help.

    The Assistant has access to a tool to write files. If the Assistant says they have written a file, then you can assume they have done so.
    Overall you should give the Assistant the benefit of the doubt if they say they've done something. But you should reject if you feel that more work should go into this.

    """
        # If evaluator already gave feedback before, warn about repeated mistakes
        if state["feedback_on_work"]:
            user_message += f"Also, note that in a prior attempt from the Assistant, you provided this feedback: {state['feedback_on_work']}\n"
            user_message += "If you're seeing the Assistant repeating the same mistakes, then consider responding that user input is required."

        evaluator_messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message),
        ]

        # Invoke evaluator — returns EvaluatorOutput Pydantic object (not raw text)
        eval_result = self.evaluator_llm_with_output.invoke(evaluator_messages)

        # Return new state with evaluator's decisions
        return {
            "messages": [{"role": "assistant", "content": f"Evaluator Feedback on this answer: {eval_result.feedback}"}],
            "feedback_on_work": eval_result.feedback,
            "success_criteria_met": eval_result.success_criteria_met,
            "user_input_needed": eval_result.user_input_needed,
        }

    # --- EVALUATION ROUTER ---
    def route_based_on_evaluation(self, state: State) -> str:
        """
        Conditional edge after evaluator:
        - Success OR user input needed → END (return to user)
        - Otherwise → "worker" (try again with feedback)
        """
        if state["success_criteria_met"] or state["user_input_needed"]:
            return "END"
        else:
            return "worker"

    # --- GRAPH BUILDING (The 5 Steps) ---
    async def build_graph(self):
        """Build the LangGraph: nodes, edges, compile with checkpointer."""
        graph_builder = StateGraph(State)

        # Step 3: Add nodes
        graph_builder.add_node("worker", self.worker)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        graph_builder.add_node("evaluator", self.evaluator)

        # Step 4: Add edges
        # Worker → tools OR evaluator (conditional)
        graph_builder.add_conditional_edges("worker", self.worker_router, {"tools": "tools", "evaluator": "evaluator"})
        # Tools → always back to worker
        graph_builder.add_edge("tools", "worker")
        # Evaluator → worker (retry) OR END (done/stuck)
        graph_builder.add_conditional_edges("evaluator", self.route_based_on_evaluation, {"worker": "worker", "END": END})
        # Entry point
        graph_builder.add_edge(START, "worker")

        # Step 5: Compile with checkpointer for memory between super steps
        self.graph = graph_builder.compile(checkpointer=self.memory)

    # --- SUPER STEP EXECUTION ---
    async def run_superstep(self, message, success_criteria, history):
        """
        Run one complete super step (one user interaction through the full graph).

        The graph may loop internally (worker ⇄ tools, worker ⇄ evaluator)
        but from the user's perspective, this is one request → one response.
        """
        config = {"configurable": {"thread_id": self.sidekick_id}}

        state = {
            "messages": message,
            "success_criteria": success_criteria or "The answer should be clear and accurate",
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
        }

        # Invoke the graph asynchronously
        result = await self.graph.ainvoke(state, config=config)

        # Package results for the UI
        user = {"role": "user", "content": message}
        reply = {"role": "assistant", "content": result["messages"][-2].content}      # Worker's final answer
        feedback = {"role": "assistant", "content": result["messages"][-1].content}    # Evaluator's feedback
        return history + [user, reply, feedback]

    # --- CLEANUP ---
    def cleanup(self):
        """Close browser and Playwright resources when session ends."""
        if self.browser:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.browser.close())
                if self.playwright:
                    loop.create_task(self.playwright.stop())
            except RuntimeError:
                asyncio.run(self.browser.close())
                if self.playwright:
                    asyncio.run(self.playwright.stop())
