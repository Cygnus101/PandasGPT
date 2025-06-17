# Safeframe-AI

A safety-first, self-healing agent that turns natural-language data-analysis questions into battle-tested **pandas** code.

---

## Author  
Rajath Rajesh

## Large-Language Model  
Gemini 2.0 Flash (via `langchain_google_genai`)  
*(easily swappable for GroqCloud, OpenAI, etc.)*

---

## What’s done so far 🚀  

| Layer | Status | Highlights |
|-------|--------|------------|
| **Meta-agent** | ✓ | Converts NL question → pandas/Matplotlib/Seaborn code|
| **Static guard (`guard.py`)** | ✓ | AST-based syntax & safety checks, blocks dangerous imports/calls, validates column names, **now permits columns created in-snippet**. |
| **Sandbox (`run_in_repl`)** | ✓ | Python-AST REPL in a background thread, 2-second timeout, `matplotlib` switched to `Agg` to avoid macOS GUI crashes. |
| **Self-healing loop** | ✓ | Guard → Sandbox; on error sends a repair prompt (incl. error text) to the Meta-agent which rewrites the code ; retries ≤ 3. |
| **Cross-checker** | ✓ | Second LLM pass judges semantic correctness; can trigger up to 2 extra repair attempts. Afterwords, one final call to cross-verify |
| **CLI (`main.py`)** | ✓ | ```python3 main.py --data <file> --query "<question>"``` works for `.txt`, `.csv`, `.parquet`, `.feather`. |
| **Dataset helpers** | ✓ | `ucl_dataset_prep` (UCI power data) + generic loader. |
| **Evaluation scaffold** | ✓ | 15 NL queries (UCI Household Energy) for quick benchmarking. |
| **Docs** | ✓ | README, notebook, prompt templates, guard comments, and usage examples. |

---

### Quick start

```bash
# create & activate venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # pandas, matplotlib, langchain, google-genai, …

# set your Gemini API key
echo "GOOGLE_API_KEY=<your key>" > .env

# run a sample query
python3 main.py --data dataset/household_power_consumption.txt \
                --query "What was the average active power consumption in March 2007?"
