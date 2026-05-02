"""
sidekick_tools.py — Tool Arsenal for the Sidekick Agent

This module defines and assembles all the tools that the Sidekick agent can use.
Tools are the agent's "superpowers" — each one gives it a new capability.

Tools included:
- Playwright browser automation (navigate, click, extract text, etc.)
- Push notifications (via Pushover API)
- File management (sandboxed to ./sandbox directory)
- Web search (via Google Serper API)
- Wikipedia lookup
- Python REPL (⚠️ unsandboxed — use with caution)
"""

from playwright.async_api import async_playwright
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from dotenv import load_dotenv
import os
import requests
from langchain.agents import Tool
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_experimental.tools import PythonREPLTool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper


# Load environment variables (.env file must contain API keys)
load_dotenv(override=True)

# Pushover configuration for push notifications
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user = os.getenv("PUSHOVER_USER")
pushover_url = "https://api.pushover.net/1/messages.json"

# Google Serper API wrapper for web search
serper = GoogleSerperAPIWrapper()


async def playwright_tools():
    """
    Launch a Playwright browser and return LangChain-compatible tools.

    Returns:
        tuple: (tools_list, browser_instance, playwright_instance)
        - tools_list: LangChain tools for browser interaction
        - browser/playwright: kept for cleanup when session ends
    """
    # Start Playwright and launch Chromium in visible (headful) mode
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False)
    # Build LangChain toolkit from the browser instance
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    # Return tools + refs needed for cleanup later
    return toolkit.get_tools(), browser, playwright


def push(text: str):
    """
    Send a push notification to the user via Pushover.

    Returns "success" so the LLM knows the action completed
    (returning None/null can confuse the agent).
    """
    requests.post(pushover_url, data={"token": pushover_token, "user": pushover_user, "message": text})
    return "success"


def get_file_tools():
    """
    Get file management tools sandboxed to the ./sandbox directory.

    The agent can read, write, list, and delete files — but ONLY within sandbox/.
    It cannot access files outside this directory.
    """
    toolkit = FileManagementToolkit(root_dir="sandbox")
    return toolkit.get_tools()


async def other_tools():
    """
    Assemble all non-Playwright tools into a single list.

    To add new tools to the Sidekick, simply append them to the return list.
    To remove dangerous tools (like Python REPL), just remove them from here.
    """
    # Custom push notification tool
    push_tool = Tool(
        name="send_push_notification",
        func=push,
        description="Use this tool when you want to send a push notification"
    )

    # File system tools (sandboxed)
    file_tools = get_file_tools()

    # Web search via Google Serper API
    tool_search = Tool(
        name="search",
        func=serper.run,
        description="Use this tool when you want to get the results of an online web search"
    )

    # Wikipedia lookup tool
    wikipedia = WikipediaAPIWrapper()
    wiki_tool = WikipediaQueryRun(api_wrapper=wikipedia)

    # ⚠️ Python REPL — runs arbitrary Python code on your machine
    # Remove this from the list below if you're not comfortable with it
    python_repl = PythonREPLTool()

    return file_tools + [push_tool, tool_search, python_repl, wiki_tool]
