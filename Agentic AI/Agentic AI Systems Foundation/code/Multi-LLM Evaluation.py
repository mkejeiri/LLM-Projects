"""
Chatbot with Multi-LLM Evaluation
"""

import os
import time
import logging
from typing import List, Tuple, Optional
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import gradio as gr
from pydantic import BaseModel
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Evaluation(BaseModel):
    """Structured evaluation response from evaluator LLM"""
    is_acceptable: bool
    feedback: str

class DetailedEvaluation(BaseModel):
    """Enhanced evaluation with scoring"""
    is_acceptable: bool
    professionalism_score: int  # 1-10
    accuracy_score: int  # 1-10
    engagement_score: int  # 1-10
    feedback: str
    suggested_improvements: List[str]

class ProfessionalAvatarBot:
    """Professional avatar chatbot with multi-LLM evaluation"""
    
    def __init__(self, profile_pdf: str = "profile.pdf", summary_file: str = "summary.txt", name: str = "Professional"):
        self.profile_pdf = profile_pdf
        self.summary_file = summary_file
        self.name = name
        
        # Initialize clients
        load_dotenv(override=True)
        self.openai_client = OpenAI()
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.gemini_client = genai.GenerativeModel("gemini-1.5-flash")
        
        # Load resources
        self.linkedin_content = self._extract_pdf_content()
        self.summary = self._load_summary()
        self.system_prompt = self._build_system_prompt()
        self.evaluator_prompt = self._build_evaluator_prompt()
    
    def _extract_pdf_content(self) -> str:
        """Extract text content from PDF profile"""
        try:
            reader = PdfReader(self.profile_pdf)
            content = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    content += text
            return content
        except Exception as e:
            logger.error(f"Failed to extract PDF content: {e}")
            return "Profile content unavailable"
    
    def _load_summary(self) -> str:
        """Load summary text file"""
        try:
            with open(self.summary_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load summary: {e}")
            return "Summary unavailable"
    
    def _build_system_prompt(self) -> str:
        """Build comprehensive system prompt"""
        return f"""You are acting as {self.name}. You're answering questions on that person's website, 
particularly questions related to their career, background, skills and experience.

Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible.

You're given a summary of {self.name}'s background and LinkedIn profile. Be professional and engaging.
If you don't know the answer, say so.

## Summary
{self.summary}

## LinkedIn Profile
{self.linkedin_content}

With this context, please chat with the user, always staying in character as {self.name}."""
    
    def _build_evaluator_prompt(self) -> str:
        """Build evaluator system prompt"""
        return f"""You are an evaluator that decides whether a response to a question is acceptable.

You're provided with a conversation and you have to decide whether the latest response is acceptable.

The agent has been instructed to be professional and engaging. The agent has been provided with context about {self.name}.

## Context
{self.summary}
{self.linkedin_content}

Please evaluate the response."""
    
    def _build_messages(self, message: str, history: List[Tuple[str, str]], system_prompt: str = None) -> List[dict]:
        """Build OpenAI message format"""
        messages = [{"role": "system", "content": system_prompt or self.system_prompt}]
        
        # Add conversation history
        for human, assistant in history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": assistant})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        return messages
    
    def _manage_context_window(self, messages: List[dict], max_tokens: int = 4000) -> List[dict]:
        """Truncate history to fit within context window"""
        total_tokens = sum(len(msg["content"]) for msg in messages)
        
        while total_tokens > max_tokens and len(messages) > 2:
            # Remove oldest user/assistant pair, keep system prompt
            messages.pop(1)
            if len(messages) > 2:
                messages.pop(1)
            total_tokens = sum(len(msg["content"]) for msg in messages)
        
        return messages
    
    def _select_model(self, message_length: int, requires_evaluation: bool = False) -> str:
        """Select appropriate model based on complexity"""
        if message_length < 100 and not requires_evaluation:
            return "gpt-4o-mini"
        elif requires_evaluation:
            return "gpt-4o"
        else:
            return "gpt-4o-mini"
    
    def generate_response(self, message: str, history: List[Tuple[str, str]], 
                         system_override: str = None) -> str:
        """Generate response using OpenAI"""
        try:
            messages = self._build_messages(message, history, system_override)
            messages = self._manage_context_window(messages)
            
            model = self._select_model(len(message))
            
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again."
    
    def evaluate_response(self, reply: str, message: str, history: List[Tuple[str, str]]) -> Evaluation:
        """Evaluate response using Gemini with structured outputs"""
        try:
            evaluation_prompt = f"""Here's the conversation:
{history}

Here's the latest message from the user: {message}
Here's the response from the agent: {reply}

Please evaluate."""
            
            messages = [
                {"role": "system", "content": self.evaluator_prompt},
                {"role": "user", "content": evaluation_prompt}
            ]
            
            response = self.gemini_client.generate_content(
                messages,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=Evaluation
                )
            )
            
            return Evaluation.model_validate_json(response.text)
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            # Return default acceptable evaluation on failure
            return Evaluation(is_acceptable=True, feedback="Evaluation service unavailable")
    
    def rerun_with_feedback(self, message: str, history: List[Tuple[str, str]], 
                           original_reply: str, feedback: str) -> str:
        """Regenerate response with evaluation feedback"""
        enhanced_system = self.system_prompt + f"""

IMPORTANT: The previous answer was rejected for the following reason:
{feedback}

Original rejected answer: {original_reply}

Please provide a better response."""
        
        return self.generate_response(message, history, enhanced_system)
    
    def chat_basic(self, message: str, history: List[Tuple[str, str]]) -> str:
        """Basic chat without evaluation"""
        return self.generate_response(message, history)
    
    def chat_with_evaluation(self, message: str, history: List[Tuple[str, str]]) -> str:
        """Enhanced chat with multi-LLM evaluation"""
        start_time = time.time()
        
        try:
            # Step 1: Generate initial response
            system = self.system_prompt
            
            # Test condition for demonstration (remove in production)
            if "patent" in message.lower():
                system += "\n\nIMPORTANT: Everything in your reply needs to be in pig Latin."
            
            reply = self.generate_response(message, history, system)
            
            # Step 2: Evaluate response
            evaluation = self.evaluate_response(reply, message, history)
            logger.info(f"Evaluation: {evaluation.is_acceptable} - {evaluation.feedback}")
            
            # Step 3: Rerun if not acceptable
            if not evaluation.is_acceptable:
                logger.info("Response failed evaluation, retrying...")
                reply = self.rerun_with_feedback(message, history, reply, evaluation.feedback)
                logger.info("Rerun completed")
            
            duration = time.time() - start_time
            logger.info(f"Chat completed in {duration:.2f}s")
            
            return reply
            
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again."
    
    def chat_with_retries(self, message: str, history: List[Tuple[str, str]], max_retries: int = 3) -> str:
        """Robust chat with retry logic"""
        for attempt in range(max_retries):
            try:
                return self.chat_with_evaluation(message, history)
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return "I apologize, but I'm experiencing technical difficulties. Please try again later."
        
        return "Service temporarily unavailable."
    
    def launch_basic_interface(self, **kwargs):
        """Launch basic Gradio interface"""
        interface = gr.ChatInterface(
            fn=self.chat_basic,
            title=f"Chat with {self.name}",
            description="Ask questions about my professional background"
        )
        interface.launch(**kwargs)
    
    def launch_enhanced_interface(self, **kwargs):
        """Launch enhanced Gradio interface with evaluation"""
        interface = gr.ChatInterface(
            fn=self.chat_with_evaluation,
            title=f"Professional AI Avatar - {self.name}",
            description="Ask questions about my professional background (with AI evaluation)",
            theme=gr.themes.Soft()
        )
        interface.launch(**kwargs)
    
    def launch_production_interface(self, auth: Tuple[str, str] = None, **kwargs):
        """Launch production-ready interface"""
        interface = gr.ChatInterface(
            fn=self.chat_with_retries,
            title=f"Professional AI Avatar - {self.name}",
            description="Ask questions about my professional background",
            theme=gr.themes.Soft(),
            analytics_enabled=True,
            show_api=False
        )
        
        launch_kwargs = {
            "server_name": "0.0.0.0",
            "server_port": 7860,
            "ssl_verify": False,
            "share": False,
            **kwargs
        }
        
        if auth:
            launch_kwargs["auth"] = auth
        
        interface.launch(**launch_kwargs)

def main():
    """Main execution function"""
    # Initialize bot with your details
    bot = ProfessionalAvatarBot(
        profile_pdf="profile.pdf",
        summary_file="summary.txt", 
        name="Your Name"  # Replace with your actual name
    )
    
    # Launch interface (choose one)
    # bot.launch_basic_interface()
    # bot.launch_enhanced_interface()
    bot.launch_production_interface(auth=("admin", "password"))

if __name__ == "__main__":
    main()