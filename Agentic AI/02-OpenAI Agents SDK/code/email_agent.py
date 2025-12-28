# EMAIL AGENT - External Integration and Communication Component
# Demonstrates function tools and third-party service integration
# Key concept: Agent-driven external actions with proper error handling

import os
from typing import Dict

import sendgrid
from sendgrid.helpers.mail import Email, Mail, Content, To
from agents import Agent, function_tool

# Custom function tool for external service integration
@function_tool
def send_email(subject: str, html_body: str) -> Dict[str, str]:
    """Send an email with the given subject and HTML body - demonstrates external API integration"""
    sg = sendgrid.SendGridAPIClient(api_key=os.environ.get("SENDGRID_API_KEY"))
    from_email = Email("kejxxx@gmail.com")  # put your verified sender here
    to_email = To("mkejxxx@gmail.com")  # put your recipient here
    content = Content("text/html", html_body)
    mail = Mail(from_email, to_email, subject, content).get()
    response = sg.client.mail.send.post(request_body=mail)
    print("Email response", response.status_code)
    return "success"

# Instructions for HTML formatting and email composition
INSTRUCTIONS = """You are able to send a nicely formatted HTML email based on a detailed report.
You will be provided with a detailed report. You should use your tool to send one email, providing the 
report converted into clean, well presented HTML with an appropriate subject line."""

# Agent configured with custom function tool for external communication
email_agent = Agent(
    name="Email agent",
    instructions=INSTRUCTIONS,
    tools=[send_email],  # Custom function tool integration
    model="gpt-4o-mini",
)
