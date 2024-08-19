# -*- coding: utf-8 -*-
"""
Sample code for running asyncio with LangChain
"""
import os
import time
import asyncio

from langchain.chat_models import AzureChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
#%% Initialize Azure OpenAI
path = r"azure-openai_api_key.txt"
with open(path) as f:
    API_key = f.readlines()[0]
    
os.environ['OPENAI_API_KEY'] = API_key
os.environ['OPENAI_API_TYPE'] = 'azure'
os.environ['OPENAI_API_VERSION'] = '2023-05-15'
os.environ['OPENAI_API_BASE'] = 'openai-base-url'

#%%
llm = AzureChatOpenAI(model_name = "gpt-35-turbo",
                      temperature = 0.9)

prompt = PromptTemplate(input_variables = ['query'],
                        template = "{query}")

llm_chain = LLMChain(llm = llm,
                     prompt = prompt)

tools = [Tool(name = 'Language Model',
              func = llm_chain.run,
              description = 'Use this tool for general purpose queries and logic.')]

agent = initialize_agent(agent = 'zero-shot-react-description',
                         tools = tools,
                         llm = llm,
                         verbose = False,
                         max_iterations = 3,
                         handle_parsing_errors = True)

res = agent("Tell me a joke about rabbits and cats")
print(res['output'])

#%% Defining esrial function
def generate_serially():
    llm = AzureChatOpenAI(model_name = "gpt-35-turbo",
                          temperature = 0.9)
    
    for _ in range(5):
        res = llm.generate(["Tell me a sentence about something interesting about Mojo coding language"])
        print(res.generations[0][0].text)
        
#%% Defining async function
async def generate_async(llm):
    res = await llm.agenerate(["Tell me a sentence about something interesting about Mojo coding language"])
    print(res.generations[0][0].text)
    
#%% Define the function that runs the async call concurrently
async def generate_concurrently():
    llm = AzureChatOpenAI(model_name = "gpt-35-turbo",
                      temperature = 0.9)
    
    tasks = [generate_async(llm) for _ in range(5)]
    
    # asyncio.gather is a function that takes a list of tasks and runs them all at once
    await asyncio.gather(*tasks)

#%% Timing serial and concurrent functions
s = time.perf_counter()
generate_serially()
elapsed = time.perf_counter() - s
print(f"\nSerial call executed in {elapsed:0.2f} seconds.")

s = time.perf_counter()
asyncio.run(generate_concurrently())
elapsed = time.perf_counter() - s
print(f"\nConcurrent call executed in {elapsed:0.2f} seconds.")