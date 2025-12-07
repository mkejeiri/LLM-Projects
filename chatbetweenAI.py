import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')



if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")


openai = OpenAI()

system_prompt_alex = """
You are Alex, a chatbot who is very argumentative; you disagree with anything in the conversation and you challenge everything, in a snarky way.
You are in a conversation with Blake and Charlie.
"""


conversation="" 

system_prompt_blake= """
You are Blake, a very polite, courteous chatbot. You try to agree with \
everything the other person says, or find common ground. If the other person is argumentative, \
you try to calm them down and keep chatting.
"""

system_prompt_charlie= """
You are Charlie, a very kirky, but rightous chatbot. You try to reconciale \
but not at the expense of his pride. if others are argumentative try to disarm their arguments in a funny way 
"""


user_prompt_alex = f"""
You are Alex, in conversation with Blake and Charlie.
The conversation so far is as follows:
{conversation}
Now with this, respond with what you would like to say next, as Alex.
"""

user_prompt_blake = f"""
You are Blake, in conversation with Alex and Charlie.
The conversation so far is as follows:
{conversation}
Now with this, respond with what you would like to say next, as Blake.
"""


user_prompt_charlie = f"""
You are Charlie, in conversation with Blake and Alex.
The conversation so far is as follows:
{conversation}
Now with this, respond with what you would like to say next, as Charlie.
"""
gpt_model_blake="gpt-4.1-mini"
gpt_model_alex = "gpt-3.5-turbo"
gpt_model_charlie="gpt-4.1-nano"
conversation=""
def call_blake():
   messages = [{"role": "system", "content": system_prompt_blake}]   
   messages.append({"role": "assistant", "content": conversation})
   messages.append({"role": "user", "content": user_prompt_blake})
   response = openai.chat.completions.create(model=gpt_model_blake, messages=messages)
   return f"""Blake reply: {response.choices[0].message.content}"""


def call_alex():
    messages = [{"role": "system", "content": system_prompt_alex}]    
    messages.append({"role": "assistant", "content": conversation})    
    messages.append({"role": "user", "content": user_prompt_alex})
    response = openai.chat.completions.create(model=gpt_model_alex, messages=messages)
    return f"""Alex reply: {response.choices[0].message.content}"""

def call_charlie():
    messages = [{"role": "system", "content": system_prompt_alex}]    
    messages.append({"role": "assistant", "content": conversation})    
    messages.append({"role": "user", "content": user_prompt_charlie})
    response = openai.chat.completions.create(model=gpt_model_charlie, messages=messages)
    return f"""Charlie reply: {response.choices[0].message.content}"""


for i in range(5):
    gpt_next = call_blake()
    display(Markdown(f"### GPT:\n{gpt_next}\n"))
    conversation = f"""{conversation}\n{gpt_next}"""
    
    gpt_next = call_alex()
    display(Markdown(f"### GPT:\n{gpt_next}\n"))
    conversation = f"""{conversation}\n{gpt_next}"""

    gpt_next = call_charlie()
    display(Markdown(f"### GPT:\n{gpt_next}\n"))
    conversation = f"""{conversation}\n{gpt_next}"""
