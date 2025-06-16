from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import json
import pandas as pd
from sandbox import run_in_repl
from guard import validate_code
from utils.prompt import load_prompt
load_dotenv()
import json
from langchain_google_genai import ChatGoogleGenerativeAI



llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

def build_cross_prompt(user_query: str,
                       dataset_info: str,
                       code: str,
                       result_snippet: str = "") -> str:
    """
    Assemble the prompt for the LLM cross-checker.

    Parameters
    ----------
    user_query : str
        Original natural-language question.
    dataset_info : str
        Schema + sample rows block (same text you fed to the generator).
    code : str
        The candidate Python code to be validated.
    result_snippet : str, optional
        Small textual representation of the sandbox result
        (e.g. a number, head() of a DataFrame). Leave empty if large.

    Returns
    -------
    str
        Fully-formed prompt string ready to send to the LLM.
    """
    result_block = (
        f"\n## Sandbox result (truncated)\n{result_snippet}\n"
        if result_snippet else ""
    )

    return (
        "You are a code-review assistant.\n\n"
        "## User question\n"
        f"{user_query}\n\n"
        "## Dataset context\n"
        f"{dataset_info}\n\n"
        "## Candidate solution code\n"
        "```python\n"
        f"{code}\n"
        "```\n"
        f"{result_block}\n"
        "### Task\n"
        "Evaluate whether the code, when executed on the full dataset, "
        "correctly answers the user’s question.\n\n"
        "Respond **only** with a JSON object of the form:\n\n"
        "{\n"
        '  "valid": true | false,\n'
        '  "reason": "<one concise sentence>",\n'
        '  "fix_hint": "<brief suggestion>"\n'
        "}\n\n"
        "Do not output any additional text outside the JSON."
    )

def cross_check(user_q: str, ctx: str, code: str, result_snippet: str | None = None):
    prompt = build_cross_prompt(user_q, ctx, code, result_snippet or "")
    response = llm.invoke(prompt)          # returns LangChain Message
    text = response.content.strip()


    # remove ```json``` or ``` fences if the model adds them
    if text.startswith("```"):
        text = text.split("```")[1].strip()
    
    if text.lower().startswith("json"):
        text = text[len("json"):].strip()  
    print(text)
    try:
        verdict = json.loads(text)
    except json.JSONDecodeError:
        verdict = {"valid": False,
                   "reason": "LLM reply not valid JSON",
                   "fix_hint": ""}

    # Ensure keys exist
    return {
        "valid": bool(verdict.get("valid")),
        "reason": verdict.get("reason", ""),
        "fix_hint": verdict.get("fix_hint", "")
    }