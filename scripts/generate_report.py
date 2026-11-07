"""Export a completed study: python scripts/generate_report.py STUDY_DIRECTORY."""

import sys

from tabular_thesis.experiments.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["report", *sys.argv[1:]]))
