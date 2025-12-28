# Import required libraries for the professional career chatbot
from dotenv import load_dotenv  # Load environment variables from .env file
from openai import OpenAI       # OpenAI API client for LLM interactions
import json                     # JSON handling for tool calls
import os                       # Operating system interface for environment variables
import requests                 # HTTP requests for Pushover notifications
from pypdf import PdfReader     # PDF reading for LinkedIn profile
import gradio as gr             # Web interface framework

# Load environment variables (API keys, tokens)
load_dotenv(override=True)

# Pushover notification function - sends push notifications to phone
# This is a simple alternative to SMS that doesn't require complex regulations
def push(text):
    """Send push notification via Pushover service"""
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),  # Pushover app token
            "user": os.getenv("PUSHOVER_USER"),    # Pushover user key
            "message": text,                        # Message to send
        }
    )


# Tool functions that the LLM can call
# These are the actual Python functions that get executed when tools are invoked

def record_user_details(email, name="Name not provided", notes="not provided"):
    """Tool function: Record user contact details when they want to get in touch"""
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    """Tool function: Record questions the AI couldn't answer for follow-up"""
    push(f"Recording {question}")
    return {"recorded": "ok"}

# JSON schema definitions for tools - this is what gets sent to OpenAI
# The LLM uses these descriptions to decide when and how to call tools
# This is the "language" that LLMs understand for tool capabilities

record_user_details_json = {
    "name": "record_user_details",  # Function name to call
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {  # Parameter schema in JSON Schema format
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            },
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],  # Only email is mandatory
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",  # Function name to call
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],  # Question text is required
        "additionalProperties": False
    }
}

# Package the tool definitions into the format OpenAI expects
# This is the complete JSON structure sent to the LLM describing available tools
tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]


# Main agent class that handles the professional career chatbot
class ProfessionalAgent:
    """Professional career chatbot that can answer questions about background and use tools"""

    def __init__(self):
        """Initialize the agent with OpenAI client and load personal data"""
        self.openai = OpenAI()  # Initialize OpenAI client
        self.name = "[Your Name]"  # Replace with actual name
        
        # Load LinkedIn profile from PDF
        reader = PdfReader("me/linkedin.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        
        # Load additional summary information
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()


    def handle_tool_call(self, tool_calls):
        """Handle tool calls from the LLM - this is the 'glorified if statement'
        
        This function takes tool calls from OpenAI and executes the corresponding Python functions.
        It's essentially mapping JSON function names to actual Python functions and calling them.
        """
        results = []
        for tool_call in tool_calls:
            # Extract function name and arguments from the tool call
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            
            # Use Python's globals() to dynamically find and call the function
            # This is the "magic" that avoids having explicit if statements
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            
            # Format the result for OpenAI's expected response format
            results.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id
            })
        return results
    
    def system_prompt(self):
        """Generate the system prompt that defines the AI's role and behavior
        
        This prompt includes:
        - Role definition (acting as the person)
        - Instructions for tool usage
        - Personal data (LinkedIn profile, summary)
        - Behavioral guidelines
        """
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

        # Include the personal data as context
        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt
    
    def chat(self, message, history):
        """Main chat function that handles the conversation loop with tool calling
        
        This is the core of the agentic behavior:
        1. Send message + tools to OpenAI
        2. If it wants to call tools, execute them and loop back
        3. Continue until it gives a final response
        
        This loop is what enables the AI to use tools and interact with the real world.
        """
        # Build the conversation context
        messages = [
            {"role": "system", "content": self.system_prompt()},  # System instructions
            *history,  # Previous conversation
            {"role": "user", "content": message}  # Current user message
        ]
        
        # Keep looping until we get a final response (not a tool call)
        done = False
        while not done:
            # Call OpenAI with tools available
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini", 
                messages=messages, 
                tools=tools  # This tells OpenAI what tools are available
            )
            
            # Check if the AI wants to call tools
            if response.choices[0].finish_reason == "tool_calls":
                # Extract and execute the tool calls
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                
                # Add the tool call and results to conversation history
                messages.append(message)
                messages.extend(results)
                # Loop back to get the final response
            else:
                # We have a final response, exit the loop
                done = True
                
        return response.choices[0].message.content
    

# Main execution - create and launch the Gradio web interface
if __name__ == "__main__":
    # Create the professional agent instance
    pa = ProfessionalAgent()
    # Launch Gradio chat interface that calls the agent's chat method
    gr.ChatInterface(pa.chat, type="messages").launch()
    