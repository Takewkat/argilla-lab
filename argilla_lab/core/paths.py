"""Where the lab keeps its files. No .env needed, so the SWE-bench audit runs without Argilla."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"
