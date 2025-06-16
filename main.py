import pandas as pd
from utils.preprocess import ucl_dataset_prep
from utils.prompt import load_prompt
from agents.meta_agent import generate_code_sequence, try_generate_and_execute
from guard import validate_code
import matplotlib.pyplot as plt
from sandbox import run_in_repl
from agents.crosschecker import cross_check
# --------------------------------------------------
# helpers
# --------------------------------------------------

def dataframe_context(df: pd.DataFrame, n_rows: int = 5) -> str:
    """Return column names + dtypes and a small sample as plain text."""
    schema = "\n".join(f"- **{c}**: {t}" for c, t in df.dtypes.items())
    sample = df.head(n_rows).to_markdown(index=False)
    return (
        "### Dataset schema\n" + schema + "\n\n" +
        "### Sample rows\n" + sample + "\n"
    )


# --------------------------------------------------
# main workflow
# --------------------------------------------------

def main() -> None:
    # 1) Load & preprocess dataset
    df = ucl_dataset_prep("dataset/household_power_consumption.txt")

    # 2) Build dataset context block for the LLM
    ctx = dataframe_context(df)

    # 3) Load meta‑agent prompt template
    template = load_prompt("prompts/meta_agent.txt")

    # 4) Natural‑language question (could be user input)
    user_query = "Give me average of first 2 columns?"

    # 5) Assemble final prompt
    prompt = (
        template
        .replace("{{DATASET_INFO}}", ctx)
        .replace("{{USER_QUERY}}", user_query)
    )

    out = try_generate_and_execute(prompt, df)

    if out["ok"]:
        # -----------------------------------------------
        #  NEW ︱ LLM cross-checker sanity pass
        # -----------------------------------------------
        verdict = cross_check(
            user_query,          # original NL question
            ctx,                 # dataset schema + sample rows
            out["code"],         # candidate code that ran successfully
        )

        if verdict["valid"]:
            print("Cross-check passed.")
            print("Final code:\n", out["code"])
            print("Result:", out["result"])
        else:
            print("Cross-check flagged an issue:")
            print("Reason :", verdict["reason"])
            print("Hint   :", verdict["fix_hint"])
            # You could launch one more repair cycle here if you wish.

    else:
        # self-healing failed after 3 tries
        print(out["error"])
        choice = input("Keep this partial code? (y/n) ")
        if choice.lower().startswith("y"):
            print(out["code"])


if __name__ == "__main__":
    main()

