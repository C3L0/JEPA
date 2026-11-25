"""Convenience wrapper; prefer `python -m ajeppa_m4` from the project root."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ajeppa_m4.cli import main

if __name__ == "__main__":
    main()
