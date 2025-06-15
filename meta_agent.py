from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import json
import pandas as pd

load_dotenv()

from utils.prompt import load_prompt
from utils.preprocess import ucl_dataset_prep

prompt_path = "prompts/meta_agent.txt"

def generate_code_sequence(prompt):
    # Initialize the Google Generative AI client
    template = load_prompt(prompt_path)
    prompt = f"{template}\nUser query: {prompt}\n\nOutput:"
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    raw = llm.invoke(prompt).content.strip()
    return raw

# print(generate_code_sequence("What was the average active power consumption in March 2007?"))





