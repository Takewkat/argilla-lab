"""Settings shared by the lab commands, read from .env at the repository root."""

import os
from pathlib import Path

from dotenv import load_dotenv

from argilla_lab.core.paths import ROOT

load_dotenv(ROOT / ".env")


def env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set, copy .env.example to .env and fill it")
    return value


API_URL = os.environ.get("ARGILLA_API_URL", "http://127.0.0.1:6900")
API_KEY = env("ARGILLA_API_KEY")
WORKSPACE = os.environ.get("ARGILLA_WORKSPACE", "default")
PRIVATE_DIR = Path(env("LAB_PRIVATE_DIR")).expanduser()
