import argparse
import os
import pandas as pd
import matplotlib       # suppress macOS GUI crashes
matplotlib.use("Agg")

from utils.preprocess import ucl_dataset_prep
from utils.prompt import load_prompt
from agents.meta_agent import try_generate_and_execute
from agents.crosschecker import cross_check
# guard, run_in_repl, etc. are imported inside the meta / sandbox layer

# ---------- helpers -------------------------------------------------

def dataframe_context(df: pd.DataFrame, n_rows: int = 5) -> str:
    schema = "\n".join(f"- **{c}**: {t}" for c, t in df.dtypes.items())
    sample = df.head(n_rows).to_markdown(index=False)
    return f"### Dataset schema\n{schema}\n\n### Sample rows\n{sample}\n"

def load_dataframe(path: str) -> pd.DataFrame:
    if path.endswith(".txt"):
        return ucl_dataset_prep(path)
    elif path.endswith((".csv", ".parquet", ".feather")):
        return pd.read_csv(path) if path.endswith(".csv") else pd.read_parquet(path)
    else:
        raise ValueError("Unsupported file type: " + os.path.basename(path))

# ---------- main ----------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PandasGPT pipeline")
    parser.add_argument("--data", required=True,
                        help="Path to dataset file (txt/csv/parquet)")
    parser.add_argument("--query", required=True,
                        help="Natural-language question for the LLM")
    args = parser.parse_args()

    # 1 ·  Load dataset
    df = load_dataframe(args.data)

    # 2 ·  Build prompt context
    ctx = dataframe_context(df)
    template = load_prompt("prompts/meta_agent.txt")
    prompt   = template.replace("{{DATASET_INFO}}", ctx) \
                       .replace("{{USER_QUERY}}", args.query)

    # 3 ·  Generate → self-heal → sandbox
    out = try_generate_and_execute(prompt, df)

    # 4 ·  Cross-check if code ran
    if out["ok"]:
        verdict = cross_check(args.query, ctx, out["code"])
        if verdict["valid"]:
            print("Answer produced by validated code:")
            print(out["result"])
        else:
            print("Cross-checker doubts the answer.")
            print("Reason :", verdict["reason"])
            print("Hint   :", verdict["fix_hint"])
            print("\nLast code attempt:\n", out["code"])
    else:
        print("Pipeline failed after retries:", out["error"])
        print("\nLast code attempt:\n", out["code"])

if __name__ == "__main__":
    main()