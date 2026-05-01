# push_tool.py - Custom tool that sends push notifications to the user via Pushover
# Demonstrates: how to build a custom CrewAI tool with Pydantic schema + _run() method
# Requires PUSHOVER_USER and PUSHOVER_TOKEN in .env

from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import os
import requests


# Pydantic model defines the JSON schema for tool arguments
# The LLM will produce JSON matching this schema when it decides to use the tool
class PushNotification(BaseModel):
    """A message to be sent to the user"""
    message: str = Field(..., description="The message to be sent to the user.")


# Custom tool class - inherits from BaseTool
class PushNotificationTool(BaseTool):
    # name: what the LLM sees as the tool identifier
    name: str = "Send a Push Notification"
    # description: tells the LLM WHEN to use this tool - critical for tool selection
    description: str = (
        "This tool is used to send a push notification to the user."
    )
    # args_schema: links the Pydantic model as the expected input format
    args_schema: Type[BaseModel] = PushNotification

    # _run(): the actual execution logic when the LLM calls this tool
    # Parameters match the fields in the Pydantic schema
    def _run(self, message: str) -> str:
        pushover_user = os.getenv("PUSHOVER_USER")
        pushover_token = os.getenv("PUSHOVER_TOKEN")
        pushover_url = "https://api.pushover.net/1/messages.json"

        print(f"Push: {message}")
        payload = {"user": pushover_user, "token": pushover_token, "message": message}
        requests.post(pushover_url, data=payload)
        return '{"notification": "ok"}'
