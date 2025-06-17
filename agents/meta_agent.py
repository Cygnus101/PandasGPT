from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import json
import pandas as pd
from sandbox import run_in_repl
from guard import validate_code
from utils.prompt import load_prompt
load_dotenv()

prompt_path = "prompts/meta_agent.txt"

def build_repair_prompt(base_prompt: str,
                        bad_code: str,
                        error_type: str,
                        error_msg: str) -> str:
    # remove any fences from bad_code / error_msg first if you like
    block = (
        "\n\n---\n"
        f"Previous attempt failed on **{error_type}**:\n"
        f"{bad_code}\n\n"
        "Error message / critic feedback:\n"
        f"{error_msg.strip()}\n\n"
        "Please rewrite the code so it works, "
        "assign the final answer to _ , "
        "and return only the Python code."
    )
    return base_prompt + block


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
    if raw.startswith("```"):
        raw = raw.split("```")[1].strip()
    
    if raw.lower().startswith("python"):
        raw = raw[len("python"):].strip()  
    return raw



MAX_TRIES = 3

def try_generate_and_execute(prompt_base: str, df):
    code = generate_code_sequence(prompt_base)         # first attempt
    for attempt in range(1, MAX_TRIES + 1):
        print(f"\nAttempt {attempt}:\n{code}")

        # ---------- static guard ----------
        verdict = validate_code(code, {"columns": df.columns})
        if not verdict["ok"]:
            error_type = "guard"
            error_msg  = "; ".join(verdict["issues"])
        else:
            # ---------- sandbox ----------
            run = run_in_repl(code, df)               # full df
            if run["ok"]:
                return {"ok": True, "code": code, "result": run["result"]}
            error_type = "sandbox"
            error_msg  = run["error"]

        # ---------- build repair prompt ----------
        repair_prompt = build_repair_prompt(
            prompt_base, code, error_type, error_msg
        )
        code = generate_code_sequence(repair_prompt)

    # exhausted retries
    return {
        "ok": False,
        "code": code,
        "error": f"Failed after {MAX_TRIES} tries ({error_type}): {error_msg}"
    }







