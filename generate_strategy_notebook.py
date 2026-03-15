from pathlib import Path

import nbformat as nbf


REPO_BOOTSTRAP = """
from pathlib import Path
import os
import sys

def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "run_trade_from_cache.py").exists() or (candidate / "run_discovery_cache.py").exists():
            return candidate
    raise RuntimeError("Could not locate the repository root from the current working directory.")

repo_root = find_repo_root(Path.cwd())
os.chdir(repo_root)
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
print(f"Repository root: {repo_root}")
""".strip()


def strategy_markdown(mode: str) -> str:
    return f"""
# Strategy Runner ({mode})

This notebook is a thin wrapper around the production trading runner.

It uses:
1. The cached discovery output in `data/latest_discovery.json`.
2. The same trade gating and execution code used by GitHub Actions.
3. A safe wrapper that treats missing credentials as a clean skip instead of a notebook crash.
""".strip()


def discovery_markdown() -> str:
    return """
# Discovery Refresh

This notebook runs the shared discovery cache refresh used by the automated workflows.

It refreshes:
1. `data/latest_discovery.json`
2. `data/discovery_history.csv`

The heavy discovery logic lives in `run_discovery_cache.py`, so this notebook stays aligned with production.
""".strip()


def safe_runner_cell(module_name: str, env_vars: dict[str, str]) -> str:
    env_lines = "\n".join(
        f"os.environ[{key!r}] = {value!r}"
        for key, value in env_vars.items()
    )
    return f"""
import os

{env_lines}

module_name = {module_name!r}
try:
    module = __import__(module_name, fromlist=["main"])
except ModuleNotFoundError as exc:
    print(f"Runner dependency missing: {{exc}}")
    module = None

if module is not None:
    try:
        module.main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise
        print(f"Runner exited cleanly with code {{exc.code}}.")
""".strip()


def create_strategy_notebook(filepath: str, mode: str):
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(strategy_markdown(mode)),
        nbf.v4.new_code_cell(REPO_BOOTSTRAP),
        nbf.v4.new_code_cell(
            safe_runner_cell(
                "run_trade_from_cache",
                {
                    "TRADING_MODE": mode,
                    "DISCOVERY_CACHE_PATH": "data/latest_discovery.json",
                    "DISCOVERY_MAX_AGE_DAYS": "10",
                },
            )
        ),
    ]

    with open(filepath, "w", encoding="utf-8") as handle:
        nbf.write(nb, handle)


def create_discovery_notebook(filepath: str):
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(discovery_markdown()),
        nbf.v4.new_code_cell(REPO_BOOTSTRAP),
        nbf.v4.new_code_cell(
            safe_runner_cell(
                "run_discovery_cache",
                {
                    "DISCOVERY_CACHE_PATH": "data/latest_discovery.json",
                },
            )
        ),
    ]

    with open(filepath, "w", encoding="utf-8") as handle:
        nbf.write(nb, handle)


def create_all_notebooks():
    create_strategy_notebook("Misc. Files/strategy-live.ipynb", "LIVE")
    create_strategy_notebook("Misc. Files/strategy-paper.ipynb", "PAPER")
    create_strategy_notebook("Misc. Files/strategy-max-paper.ipynb", "MAX-PAPER")
    create_discovery_notebook("Misc. Files/discovery-refresh.ipynb")


if __name__ == "__main__":
    create_all_notebooks()
