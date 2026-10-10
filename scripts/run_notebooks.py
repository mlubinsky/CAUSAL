"""Execute every notebook and report which ones pass.

Each notebook ends its library-check cells with assertions, so "executes without
error" == "from-scratch implementation agrees with the reference library".

    python scripts/run_notebooks.py            # run all, leave files untouched
    python scripts/run_notebooks.py --inplace  # also save outputs into the notebooks
    python scripts/run_notebooks.py 04 06      # run only notebooks whose name starts with 04 or 06
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="*", help="only run notebooks starting with these prefixes")
    ap.add_argument("--inplace", action="store_true", help="write executed outputs back to the files")
    ap.add_argument("--timeout", type=int, default=900, help="per-cell timeout in seconds")
    args = ap.parse_args()

    paths = sorted(NB_DIR.glob("[0-9][0-9]_*.ipynb"))
    if args.prefixes:
        paths = [p for p in paths if p.name.startswith(tuple(args.prefixes))]

    failures = []
    for path in paths:
        nb = nbformat.read(path, as_version=4)
        start = time.time()
        try:
            NotebookClient(nb, timeout=args.timeout, kernel_name="python3",
                           resources={"metadata": {"path": str(NB_DIR)}}).execute()
            status = "ok"
            if args.inplace:
                nbformat.write(nb, path)
        except CellExecutionError as exc:
            status = "FAILED"
            failures.append((path.name, str(exc).splitlines()[-1]))
        print(f"{status:7s} {path.name:45s} {time.time() - start:6.1f}s", flush=True)

    for name, msg in failures:
        print(f"\n{name}: {msg}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
