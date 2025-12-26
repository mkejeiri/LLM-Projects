#!/usr/bin/env python3
"""
Multi-Model API Integration with Judging and Retry Logic for question asked to different LLM with evaluation
"""

import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from IPython.display import Markdown, display

def setup_environment():
    """Load environment variables and validate all API keys"""
    load_dotenv(override=True)
    
    # Get all API keys
    openai_api_key = os.getenv('OPENAI_API_KEY')
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    google_api_key = os.getenv('GOOGLE_API_KEY')
    deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
    groq_api_key = os.getenv('GROQ_API_KEY')
    
    # Print key status
    if openai_api_key:
        print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
    else:
        print("OpenAI API Key not set")
        
    if anthropic_api_key:
        print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
    else:
        print("Anthropic API Key not set (and this is optional)")

    if google_api_key:
        print(f"Google API Key exists and begins {google_api_key[:2]}")
    else:
        print("Google API Key not set (and this is optional)")

    if deepseek_api_key:
        print(f"DeepSeek API Key exists and begins {deepseek_api_key[:3]}")
    else:
        print("DeepSeek API Key not set (and this is optional)")

    if groq_api_key:
        print(f"Groq API Key exists and begins {groq_api_key[:4]}")
    else:
        print("Groq API Key not set (and this is optional)")
    
    return {
        'openai': openai_api_key,
        'anthropic': anthropic_api_key,
        'google': google_api_key,
        'deepseek': deepseek_api_key,
        'groq': groq_api_key
    }

def generate_test_question():
    """Generate a challenging question for model comparison"""
    openai = OpenAI()
    
    request = "Please come up with a challenging, nuanced question that I can ask a number of LLMs to evaluate their intelligence. "
    request += "Answer only with the question, no explanation."
    messages = [{"role": "user", "content": request}]
    
    response = openai.chat.completions.create(
        model="gpt-5-mini",
        messages=messages,
    )
    question = response.choices[0].message.content
    print(f"Generated question: {question}")
    return question

def robust_api_call(client, model, messages, max_retries=3, is_anthropic=False):
    """Make API call with retry logic and error handling"""
    for attempt in range(max_retries):
        try:
            if is_anthropic:
                response = client.messages.create(
                    model=model, 
                    messages=messages, 
                    max_tokens=1000
                )
                return response.content[0].text
            else:
                response = client.chat.completions.create(
                    model=model, 
                    messages=messages
                )
                return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {e}")
                return None
            print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
            time.sleep(2 ** attempt)  # Exponential backoff
    return None

def test_openai_models(question):
    """Test OpenAI models with retry logic"""
    openai = OpenAI()
    messages = [{"role": "user", "content": question}]
    
    model_name = "gpt-5-nano"
    
    answer = robust_api_call(openai, model_name, messages)
    
    if answer:
        print(f"\n=== {model_name} ===")        
        return model_name, answer
    else:
        print(f"Failed to get response from {model_name}")
        return None, None

def test_anthropic_claude(question):
    """Test Anthropic Claude with retry logic"""
    claude = Anthropic()
    messages = [{"role": "user", "content": question}]
    
    model_name = "claude-sonnet-4-5"
    
    answer = robust_api_call(claude, model_name, messages, is_anthropic=True)
    
    if answer:
        print(f"\n=== {model_name} ===")
        display(Markdown(answer))
        return model_name, answer
    else:
        print(f"Failed to get response from {model_name}")
        return None, None

def test_google_gemini(question, google_api_key):
    """Test Google Gemini with retry logic"""
    gemini = OpenAI(api_key=google_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    messages = [{"role": "user", "content": question}]
    
    model_name = "gemini-2.5-flash"
    
    answer = robust_api_call(gemini, model_name, messages)
    
    if answer:
        print(f"\n=== {model_name} ===")
        display(Markdown(answer))
        return model_name, answer
    else:
        print(f"Failed to get response from {model_name}")
        return None, None

def test_deepseek(question, deepseek_api_key):
    """Test DeepSeek with retry logic"""
    deepseek = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com/v1")
    messages = [{"role": "user", "content": question}]
    
    model_name = "deepseek-chat"
    
    answer = robust_api_call(deepseek, model_name, messages)
    
    if answer:
        print(f"\n=== {model_name} ===")
        display(Markdown(answer))
        return model_name, answer
    else:
        print(f"Failed to get response from {model_name}")
        return None, None

def test_groq(question, groq_api_key):
    """Test Groq with OpenAI open source model and retry logic"""
    groq = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
    messages = [{"role": "user", "content": question}]
    
    model_name = "openai/gpt-oss-120b"
    
    answer = robust_api_call(groq, model_name, messages)
    
    if answer:
        print(f"\n=== {model_name} ===")
        display(Markdown(answer))
        return model_name, answer
    else:
        print(f"Failed to get response from {model_name}")
        return None, None

def test_ollama(question, model_name="llama2"):
    """Test Ollama local model with retry logic"""
    try:
        ollama = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"  # Required but not used
        )
        messages = [{"role": "user", "content": question}]
        
        answer = robust_api_call(ollama, model_name, messages)
        
        if answer:
            print(f"\n=== {model_name} (Ollama) ===")
            display(Markdown(answer))
            return f"{model_name} (Ollama)", answer
        else:
            print(f"Failed to get response from Ollama {model_name}")
            return None, None
    except Exception as e:
        print(f"Ollama not available: {e}")
        return None, None

def judge_responses(question, competitors, answers):
    """Use an LLM to judge and rank the responses"""
    if len(answers) < 2:
        print("Need at least 2 responses to judge")
        return
    
    openai = OpenAI()
    
    for i, (competitor, answer) in enumerate(zip(competitors, answers)):
        together += f"Response {i+1} ({competitor}):\n{answer}\n\n"
    # Create judging prompt
    judge_prompt = f"""You are judging a competition between {len(competitors)} competitors.
    Each model has been given this question:

    {question}

    Your job is to evaluate each response for clarity and strength of argument, and rank them in order of best to worst.
    Respond with JSON, and only JSON, with the following format:
    {{"results": ["best competitor number", "second best competitor number", "third best competitor number", ...]}}

    Here are the responses from each competitor:

    {together}

    Now respond with the JSON with the ranked order of the competitors, nothing else. Do not include markdown formatting or code blocks."""   
    judge_prompt += "Please evaluate and rank the following responses from different Competitors. "
    judge_prompt += "Consider accuracy, depth, clarity, and overall quality. "
    judge_prompt += "Provide a ranking from best to worst with brief explanations.\n\n"            
    
    
    messages = [{"role": "user", "content": judge_prompt}]
    
    judgment = robust_api_call(openai, "gpt-4.1-mini", messages)
    
    if judgment:
        print("\n=== JUDGMENT ===")
        display(Markdown(judgment))
        return judgment
    else:
        print("Failed to get judgment")
        return None

def save_results(question, competitors, answers, judgment=None):
    """Save results to JSON file"""
    results = {
        "question": question,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": []
    }
    
    for competitor, answer in zip(competitors, answers):
        results["results"].append({
            "model": competitor,
            "answer": answer
        })
    
    if judgment:
        results["judgment"] = judgment
    
    filename = f"model_comparison_{int(time.time())}.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {filename}")
    except Exception as e:
        print(f"Failed to save results: {e}")

def compare_all_models(include_judgment=True, save_to_file=True):
    """Compare responses from all available models with optional judging"""
    print("=== Multi-Model API Integration with Judging ===")
    
    # Setup environment
    api_keys = setup_environment()
    
    # Generate test question
    question = generate_test_question()
    
    competitors = []
    answers = []
    
    # Test all available models
    model_tests = [
        ("openai", lambda: test_openai_models(question)),
        ("anthropic", lambda: test_anthropic_claude(question)),
        ("google", lambda: test_google_gemini(question, api_keys['google'])),
        ("deepseek", lambda: test_deepseek(question, api_keys['deepseek'])),
        ("groq", lambda: test_groq(question, api_keys['groq'])),
        ("ollama", lambda: test_ollama(question))
    ]
    
    for api_name, test_func in model_tests:
        if api_name == "ollama" or api_keys.get(api_name):
            try:
                model, answer = test_func()
                if model and answer:
                    competitors.append(model)
                    answers.append(answer)
            except Exception as e:
                print(f"{api_name.title()} error: {e}")
    
    # Judge responses if requested and we have multiple responses
    judgment = None
    if include_judgment and len(answers) > 1:
        judgment = judge_responses(question, competitors, answers)
    
    # Save results if requested
    if save_to_file:
        save_results(question, competitors, answers, judgment)
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Question: {question}")
    print(f"Models tested: {len(competitors)}")
    for i, model in enumerate(competitors):
        print(f"{i+1}. {model}")
    
    return {
        "question": question,
        "competitors": competitors,
        "answers": answers,
        "judgment": judgment
    }

def run_custom_comparison(custom_question, models_to_test=None):
    """Run comparison with custom question and specific models"""
    api_keys = setup_environment()
    
    if models_to_test is None:
        models_to_test = ["openai", "anthropic", "google", "deepseek", "groq"]
    
    competitors = []
    answers = []
    
    test_functions = {
        "openai": lambda: test_openai_models(custom_question),
        "anthropic": lambda: test_anthropic_claude(custom_question),
        "google": lambda: test_google_gemini(custom_question, api_keys['google']),
        "deepseek": lambda: test_deepseek(custom_question, api_keys['deepseek']),
        "groq": lambda: test_groq(custom_question, api_keys['groq']),
        "ollama": lambda: test_ollama(custom_question)
    }
    
    for model_name in models_to_test:
        if model_name in test_functions and (model_name == "ollama" or api_keys.get(model_name)):
            try:
                model, answer = test_functions[model_name]()
                if model and answer:
                    display(Markdown(answer))
                    competitors.append(model)
                    answers.append(answer)
            except Exception as e:
                print(f"{model_name.title()} error: {e}")
    
    # Judge responses
    judgment = None
    if len(answers) > 1:
        judgment = judge_responses(custom_question, competitors, answers)
    
    return {
        "question": custom_question,
        "competitors": competitors,
        "answers": answers,
        "judgment": judgment
    }

def main():
    """Main execution function"""
    # Run full comparison with judging
    results = compare_all_models(include_judgment=True, save_to_file=True)
    
    # Example of custom comparison
    # custom_results = run_custom_comparison(
    #     "Explain the concept of consciousness in 3 sentences.",
    #     models_to_test=["openai", "anthropic"]
    # )

if __name__ == "__main__":
    main()