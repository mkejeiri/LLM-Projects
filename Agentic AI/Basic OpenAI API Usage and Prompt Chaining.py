#!/usr/bin/env python3
"""
Basic OpenAI API Usage and Prompt Chaining
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from IPython.display import Markdown, display

def setup_environment():
    """Load environment variables and validate API key"""
    load_dotenv(override=True)
    
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if openai_api_key:
        print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
        return True
    else:
        print("OpenAI API Key not set - please head to the troubleshooting guide in the setup folder")
        return False

def basic_llm_call():
    """Basic LLM API call example"""
    openai = OpenAI()
    
    messages = [{"role": "user", "content": "What is 2+2?"}]
    
    response = openai.chat.completions.create(
        model="gpt-4.1-nano",
        messages=messages
    )
    
    print(response.choices[0].message.content)

def generate_question():
    """Generate a challenging question using LLM"""
    openai = OpenAI()
    
    question = "Please propose a hard, challenging question to assess someone's IQ. Respond only with the question."
    messages = [{"role": "user", "content": question}]
    
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages
    )
    
    question = response.choices[0].message.content
    print(question)
    return question

def answer_question(question):
    """Answer the generated question"""
    openai = OpenAI()
    
    messages = [{"role": "user", "content": question}]
    
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages
    )
    
    answer = response.choices[0].message.content
    print(answer)
    display(Markdown(answer))

def commercial_application_exercise():
    """
    Commercial application exercise - prompt chaining example:
    1. Pick a business sector
    2. Identify pain points
    3. Propose AI solution
    """
    openai = OpenAI()
    
    # Step 1: Business sector selection
    sector_prompt = "Please propose a business area that might be worth exploring for an Agentic AI opportunity. Respond only with the business area."
    messages = [{"role": "user", "content": sector_prompt}]
    
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages
    )
    business_sector = response.choices[0].message.content
    print(f"Business Sector: {business_sector}")
    
    # Step 2: Pain point identification
    pain_point_prompt = f"Present a specific pain-point in the {business_sector} industry - something challenging that might be ripe for an Agentic AI solution. Respond only with the pain point description."
    messages = [{"role": "user", "content": pain_point_prompt}]
    
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages
    )
    pain_point = response.choices[0].message.content
    print(f"Pain Point: {pain_point}")
    
    # Step 3: Solution proposal
    solution_prompt = f"Given this pain point in {business_sector}: '{pain_point}', propose a specific Agentic AI solution that could address this challenge."
    messages = [{"role": "user", "content": solution_prompt}]
    
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages
    )
    solution = response.choices[0].message.content
    print(f"Proposed Solution: {solution}")
    
    return {
        "business_sector": business_sector,
        "pain_point": pain_point,
        "solution": solution
    }

def main():
    """Main execution function"""
    print("Basic OpenAI API Usage ===")
    
    # Setup environment
    if not setup_environment():
        return
    
    # Basic LLM call
    print("\n1. Basic LLM Call:")
    basic_llm_call()
    
    # Generate and answer question
    print("\n2. Question Generation and Answering:")
    question = generate_question()
    print("\nAnswer:")
    answer_question(question)
    
    # Commercial application exercise
    print("\n3. Commercial Application Exercise:")
    result = commercial_application_exercise()

if __name__ == "__main__":
    main()