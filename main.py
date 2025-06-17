import argparse
import os
import pandas as pd
import matplotlib           # use a non-GUI backend for headless safety
matplotlib.use("Agg")

from utils.preprocess import ucl_dataset_prep
from utils.prompt import load_prompt
from agents.meta_agent import try_generate_and_execute        # guard+sandbox loop
from agents.crosschecker import cross_check, repair_with_critic

# ───────────────────────── helpers ──────────────────────────

def dataframe_context(df: pd.DataFrame, n_rows: int = 5) -> str:
    """Return schema + n sample rows as markdown-flavoured text."""
    schema = "\n".join(f"- **{c}**: {t}" for c, t in df.dtypes.items())
    sample = df.head(n_rows).to_markdown(index=False)
    return f"### Dataset schema\n{schema}\n\n### Sample rows\n{sample}\n"

def load_dataframe(path: str) -> pd.DataFrame:
    """Dispatch loader by file extension."""
    if path.endswith(".txt"):
        return ucl_dataset_prep(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".feather"):
        return pd.read_feather(path)
    raise ValueError("Unsupported file type: " + os.path.basename(path))

# ───────────────────────── pipeline ─────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SafeFrame-AI CLI")
    parser.add_argument("--data",  required=True,
                        help="Dataset path (.txt/.csv/.parquet/.feather)")
    parser.add_argument("--query", required=True,
                        help="Natural-language question for the LLM")
    args = parser.parse_args()

    # 1. Load dataset
    df = load_dataframe(args.data)

    # 2. Build prompt context
    ctx       = dataframe_context(df)
    template  = load_prompt("prompts/meta_agent.txt")
    base_prompt = (template
                   .replace("{{DATASET_INFO}}", ctx)
                   .replace("{{USER_QUERY}}", args.query))

    # 3. Generate → guard/sandbox loop → critic self-healing
    out = repair_with_critic(base_prompt, df, ctx, args.query)

    # 4. Handle outcome
    # if not out["ok"]:
    #     print("\n Pipeline failed:", out["error"])
    #     print("\nLast code attempt:\n", out["code"])
    #     return

    # # (optional extra critic pass; kept for completeness)
    # verdict = cross_check(args.query, ctx, out["code"])
    # if not verdict["valid"]:
    #     print("\n  Cross-checker still doubts the answer:")
    #     print("Reason :", verdict["reason"])
    #     print("Hint   :", verdict["fix_hint"])
    #     print("\nLast code attempt:\n", out["code"])
    #     return

    # # 5. Success!
    # print("\n Answer (validated):")

    print(out["result"])
    print("\nGenerated code:\n", out["code"])


if __name__ == "__main__":
    main()