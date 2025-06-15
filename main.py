import pandas as pd
from utils.preprocess import ucl_dataset_prep
from utils.prompt import load_prompt
from agents.meta_agent import generate_code_sequence
from guard import validate_code
import matplotlib.pyplot as plt
from sandbox import run_in_repl

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
    user_query = "Give me plot of each column versus date?"

    # 5) Assemble final prompt
    prompt = (
        template
        .replace("{{DATASET_INFO}}", ctx)
        .replace("{{USER_QUERY}}", user_query)
    )

    # 6) Generate code sequence via meta‑agent
    code_sequence = generate_code_sequence(prompt)
    print("\nGenerated code:\n", code_sequence)

    # 7) Run static guard validation
    df_meta = {
        "columns": df.columns.tolist(),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }
    verdict = validate_code(code_sequence, df_meta)

    if verdict["ok"]:
        print("Guard passed – code is safe to execute.")
        try:
            # 8) Execute code in REPL with access to the DataFrame
            result = run_in_repl(code_sequence, df)
            if result["ok"]:
                print("Sandbox trial passed:")
            else:
                print("Sandbox trial failed with error:", result["error"])
        except Exception as exc:
            print("Runtime error after guard:\n", exc)

    else:
        print("Guard found issues:")
        for issue in verdict["issues"]:
            print(" •", issue)


if __name__ == "__main__":
    main()

