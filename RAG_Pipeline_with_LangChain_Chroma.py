#load lib and env var
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

##chatgpt alternative
#from langchain_ollama import ChatOllama
#from langchain_anthropic import ChatAnthropic

from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
import gradio as gr


MODEL = "gpt-4.1-nano"
DB_NAME = "vector_db"
load_dotenv(override=True)

#LLM encoder from hugging face
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#define an instance of the vector db
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)


#Retrivel from langchain hugging face
retriever = vectorstore.as_retriever()
#retriever.invoke("Who is Avery?")

### a langchain wrapper/abstraction around chromer 
##temperature  controls which tokens get selected during inference :
##temperature=0 means: always select the token with highest probability, a temperature of 0 doesn't mean outputs will always be reproducible. You also need to set a random seed and even then, it's not always reproducible.
##temperature=1 usually means: a token with 10% probability should be picked 10% of the time
llm = ChatOpenAI(temperature=0, model_name=MODEL)
#llm.invoke("Who is Avery?")



SYSTEM_PROMPT_TEMPLATE = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""


def answer_question(question: str, history):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=question)])
    return response.content
    
#answer_question("Who is Averi Lancaster?", [])    

#call visual interface
gr.ChatInterface(answer_question).launch()