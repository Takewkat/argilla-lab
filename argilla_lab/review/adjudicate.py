"""Submit a reviewer's final labels on the disputed records, next to the annotator's own.

Decisions come from adjudication.json in the reviewer-only folder; those for records not in
the dataset are skipped. The reviewer's answer is final, so its confidence is sure.

Usage: uv run lab adjudicate [dataset], code-preference-v3 by default.
"""

import json
import sys

import argilla as rg

from argilla_lab.core.config import API_KEY, API_URL, PRIVATE_DIR, WORKSPACE, env
from argilla_lab.annotate.tasks import TASKS

DATASET = sys.argv[1] if len(sys.argv) > 1 else "code-preference-v3"
REVIEWER = "reviewer"
REVIEWER_PASSWORD = env("REVIEWER_PASSWORD")

DECISIONS = json.loads((PRIVATE_DIR / "adjudication.json").read_text())

client = rg.Argilla(api_url=API_URL, api_key=API_KEY)
workspace = client.workspaces(WORKSPACE)
reviewer = client.users(REVIEWER)
if reviewer is None:
    reviewer = rg.User(username=REVIEWER, password=REVIEWER_PASSWORD, role="annotator", client=client).create()
if REVIEWER not in [u.username for u in workspace.users]:
    workspace.add_user(reviewer)

dataset = client.datasets(DATASET, workspace=WORKSPACE)
if dataset is None:
    raise SystemExit(f"{DATASET} not found in workspace {WORKSPACE}")
fields = {task["id"]: {k: task[k] for k in ("prompt", "response_a", "response_b")} for task in TASKS}
existing = {record.id: record for record in dataset.records(with_responses=True)}
decisions = [d for d in DECISIONS if d["id"] in existing]
if not decisions:
    print(f"no decision in adjudication.json is about a record of {DATASET}, nothing to adjudicate")
    raise SystemExit(0)
records = []
for decision in decisions:
    record = existing[decision["id"]]
    kept = [r for r in record.responses if r.user_id != reviewer.id]
    added = [
        rg.Response(question, decision[question], user_id=reviewer.id, status="submitted")
        for question in ("preference", "issues", "rationale")
    ] + [rg.Response("confidence", "sure", user_id=reviewer.id, status="submitted")]
    records.append(rg.Record(id=decision["id"], fields=fields[decision["id"]], responses=kept + added))
dataset.records.log(records)
print(f"{REVIEWER} adjudicated {len(decisions)} records of {DATASET}")
