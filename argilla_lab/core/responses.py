"""Load the submitted answers of an Argilla dataset: one row per annotator and record.

Which answers to read comes from the dataset's rubric, and a gold question gets its
gold label next to each answer, as <question>_gold. Shared by the gate, the report and
the export, so all three read the same rows.
"""

import hashlib
import json

import argilla as rg
import pandas as pd

from argilla_lab.core.config import API_KEY, API_URL, PRIVATE_DIR, WORKSPACE

REVIEWER = "reviewer"


def load_responses(dataset_name, rubric):
    """Submitted annotator answers of a dataset, the reviewer's excluded, and the ids of all its records.

    The record ids come from the dataset, not from the answers, so a record nobody answered still shows up.
    """
    client = rg.Argilla(api_url=API_URL, api_key=API_KEY)
    usernames = {u.id: u.username for u in client.users}
    dataset = client.datasets(dataset_name, workspace=WORKSPACE)
    if dataset is None:
        raise SystemExit(f"{dataset_name} not found in workspace {WORKSPACE}")

    # One row per (record, annotator): Argilla stores one response per question, merge them
    columns = rubric.answer_columns
    rows = {}
    record_ids = []
    for record in dataset.records(with_responses=True):
        record_ids.append(record.id)
        for question in columns:
            for response in record.responses[question]:
                if response.status != "submitted":
                    continue
                row = rows.setdefault(
                    (record.id, response.user_id),
                    {"record_id": record.id, "annotator": usernames[response.user_id]},
                )
                row[question] = response.value

    df = pd.DataFrame(list(rows.values()), columns=["record_id", "annotator", *columns])
    df = df[df["annotator"] != REVIEWER].reset_index(drop=True)
    gold = rubric.gold_question
    if gold:
        key = json.loads((PRIVATE_DIR / "answer_key.json").read_text())
        # A record missing from the key becomes NaN here and fails the gold check of the gate
        df[f"{gold.name}_gold"] = df["record_id"].map({rid: answer.get(gold.name) for rid, answer in key.items()})
    return df, sorted(record_ids)


def fingerprint(df, rubric):
    """A short hash of who answered what, to tell whether the answers changed since the gate ran."""
    columns = ["record_id", "annotator", *rubric.answer_columns]
    rows = df[columns].sort_values(["record_id", "annotator"]).to_csv(index=False)
    return hashlib.sha256(rows.encode()).hexdigest()[:12]
