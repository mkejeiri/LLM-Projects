import os
import glob
from dotenv import load_dotenv
from pathlib import Path
import gradio as gr
from openai import OpenAI

#load token
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

MODEL = "gpt-4.1-nano"
openai = OpenAI()

knowledge = {}

filenames = glob.glob("knowledge-base/employees/*")

for filename in filenames:
    #getfilename
    name = Path(filename).stem.split(' ')[-1]    
    with open(filename, "r", encoding="utf-8") as f:
        knowledge[name.lower()] = f.read()
        
#knowledge["lancaster"]
filenames = glob.glob("knowledge-base/products/*")

for filename in filenames:
    name = Path(filename).stem
    with open(filename, "r", encoding="utf-8") as f:
        knowledge[n

#knowledge.keys()        

def get_relevant_context_simple(message):
    text = ''.join(ch for ch in message if ch.isalpha() or ch.isspace())
    words = text.lower().split()
    relevant_context = []
    for word in words:
        if word in knowledge:           
            relevant_context.append(knowledge[word])
    return relevant_context          
    
    
## But a more pythonic way:
def get_relevant_context(message):
    text = ''.join(ch for ch in message if ch.isalpha() or ch.isspace())
    words = text.lower().split()
    relevant_context = []
    return [knowledge[word] for word in words if word in knowledge]       
    
  
#get_relevant_context_simple("Who is     lancaster?")
#knowledge["lancaster"]
#get_relevant_context("Who is rodriguez Lancaster and what is carllm?")

def additional_context(message):
    relevant_context = get_relevant_context(message)
    if not relevant_context:
        result = "There is no additional context relevant to the user's question."
    else:
        result = "The following additional context might be relevant in answering the user's question:\n\n"
        result += "\n\n".join(relevant_context)
    return result

#print(additional_context("Who is rodriguez Lancaster and what is carllm?"))

def chat(message, history):
    system_message = SYSTEM_PREFIX + additional_context(message)
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content
    
view = gr.ChatInterface(chat, type="messages").launch(inbrowser=True)
    