import pandas as pd
from utils.preprocess import ucl_dataset_prep
from utils.prompt import load_prompt
from meta_agent import generate_code_sequence

# ---------- helpers ----------
def dataframe_context(df: pd.DataFrame, n_rows: int = 5) -> str:
    """Return column names + dtypes and a small sample as plain text."""
    schema = "\n".join(f"- **{c}**: {t}" for c, t in df.dtypes.items())
    sample = df.head(n_rows).to_markdown(index=False)
    return f"""### Dataset schema
                {schema}

            ### Sample rows
            {sample}
            """

# ---------- main workflow ----------
def main():
    # 1. Load & preprocess dataset
    df = ucl_dataset_prep("dataset/household_power_consumption.txt")

    # 2. Build dataset context block
    ctx = dataframe_context(df)

    # 3. Load prompt template (the one you just refined)
    template = load_prompt("prompts/meta_agent.txt")

    # 4. Natural-language question
    user_query = "What was the average active power consumption in March 2007?"

    # 5. Assemble final prompt
    prompt = (template
              .replace("{{DATASET_INFO}}", ctx)
              .replace("{{USER_QUERY}}", user_query))

    # 6. Generate code sequence
    code_sequence = generate_code_sequence(prompt)
    


if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     main()

# Example usage of the ucl_dataset_prep function
# Uncomment the following lines to run the preprocessing function directly
# df = ucl_dataset_prep("dataset/household_power_consumption.txt")
# print(df.head())
# print(str(df.dtypes))