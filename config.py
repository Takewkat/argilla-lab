"""Settings shared by the lab scripts, read from .env next to this file."""

import os
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")


def env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set, copy .env.example to .env and fill it")
    return value


API_URL = os.environ.get("ARGILLA_API_URL", "http://127.0.0.1:6900")
API_KEY = env("ARGILLA_API_KEY")
WORKSPACE = os.environ.get("ARGILLA_WORKSPACE", "default")
PRIVATE_DIR = Path(env("LAB_PRIVATE_DIR")).expanduser()
