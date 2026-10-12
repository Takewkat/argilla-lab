"""Load the submitted answers of a dataset: one row per annotator and record, the gold answer next to each.

Shared by quality_gate.py and bias_report.py, so both read the same rows.
"""

import json

import argilla as rg
import pandas as pd

from config import API_KEY, API_URL, PRIVATE_DIR, WORKSPACE

QUESTIONS = ["preference", "confidence", "rationale", "issues"]
PREFERENCES = ["A", "B", "tie", "both_bad"]
SUBSTANTIVE = {"bug", "security", "ignores_instructions", "performance"}
REVIEWER = "reviewer"


def issues_kind(issues):
    """What the issue tags claim: empty, none, style_only, or substantive."""
    tags = set(issues) if isinstance(issues, list) else set()
    if not tags:
        return "empty"
    if tags & SUBSTANTIVE:
        return "substantive"
    return "none" if tags == {"none"} else "style_only"


def load_responses(dataset_name):
    """Submitted annotator answers of a dataset, the reviewer's excluded, and the ids of all its records.

    The record ids come from the dataset, not from the answers, so a record nobody answered still shows up.
    """
    key = json.loads((PRIVATE_DIR / "answer_key.json").read_text())
    gold = {record_id: answer["preference"] for record_id, answer in key.items()}
    client = rg.Argilla(api_url=API_URL, api_key=API_KEY)
    usernames = {u.id: u.username for u in client.users}
    dataset = client.datasets(dataset_name, workspace=WORKSPACE)
    if dataset is None:
        raise SystemExit(f"{dataset_name} not found in workspace {WORKSPACE}")

    # One row per (record, annotator): Argilla stores one response per question, merge them
    rows = {}
    record_ids = []
    for record in dataset.records(with_responses=True):
        record_ids.append(record.id)
        for question in QUESTIONS:
            for response in record.responses[question]:
                if response.status != "submitted":
                    continue
                row = rows.setdefault(
                    (record.id, response.user_id),
                    {"record_id": record.id, "annotator": usernames[response.user_id]},
                )
                row[question] = response.value

    df = pd.DataFrame(list(rows.values()), columns=["record_id", "annotator", *QUESTIONS])
    df = df[df["annotator"] != REVIEWER].reset_index(drop=True)
    # A record missing from the key becomes NaN here and fails the gold check of the gate
    df["key"] = df["record_id"].map(gold)
    df["correct"] = df["preference"] == df["key"]
    df["issues_kind"] = df["issues"].map(issues_kind)
    return df, sorted(record_ids)
