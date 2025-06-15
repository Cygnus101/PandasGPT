from langchain_experimental.tools.python.tool import PythonAstREPLTool
from concurrent.futures import ThreadPoolExecutor, TimeoutError

def run_in_repl(code_str: str, df):
    repl = PythonAstREPLTool(locals={"df": df})

    def _exec():
        # run() returns only printed output; capture result from repl.globals
        repl.run(code_str)
        return repl.globals.get("_")  # convention: code assigns result to _

    with ThreadPoolExecutor() as exe:
        fut = exe.submit(_exec)
        try:
            return {"ok": True, "result": fut.result(timeout=2)}
        except TimeoutError:
            return {"ok": False, "error": "Timeout"}
        except Exception as e:
            return {"ok": False, "error": str(e)}