"""Submit the four records iana left open, through the SDK instead of the UI."""

import argilla as rg

from config import API_KEY, API_URL, WORKSPACE
from tasks import TASKS

DATASET = "code-preference"
ANNOTATOR = "iana"

RESPONSES = [
    {
        "id": "t01",
        "preference": "A",
        "issues": ["performance", "style_only"],
        "rationale": "[Claude] Both reverse correctly. B prepends in a loop, O(n^2), and has no type hints.",
    },
    {
        "id": "t03",
        "preference": "A",
        "issues": ["security"],
        "rationale": "[Claude] B puts the username into the SQL string with an f-string: SQL injection. A binds it as a parameter.",
    },
    {
        "id": "t05",
        "preference": "A",
        "issues": ["bug"],
        "rationale": "[Claude] B uses bucket=[] as a default: the same list is shared across calls, so items leak between them.",
    },
    {
        "id": "t06",
        "preference": "B",
        "issues": ["bug"],
        "rationale": "[Claude] A loops forever on any exception, with no timeout and no backoff. B is bounded, though it still misses ReadTimeout.",
    },
]

client = rg.Argilla(api_url=API_URL, api_key=API_KEY)
dataset = client.datasets(DATASET, workspace=WORKSPACE)
user_id = client.users(ANNOTATOR).id
fields = {task["id"]: {k: task[k] for k in ("prompt", "response_a", "response_b")} for task in TASKS}
records = [
    rg.Record(
        id=r["id"],
        fields=fields[r["id"]],
        responses=[
            rg.Response(question, r[question], user_id=user_id, status="submitted")
            for question in ("preference", "issues", "rationale")
        ],
    )
    for r in RESPONSES
]
dataset.records.log(records)
print(f"submitted {len(RESPONSES)} responses as {ANNOTATOR}")
