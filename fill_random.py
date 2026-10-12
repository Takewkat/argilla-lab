"""Submit random preferences on code-preference-v3 for annotator2 and annotator3, through the SDK.

Placeholder annotators: their votes carry no judgement, so the kappa they produce says nothing.
"""

import random

import argilla as rg

from config import API_KEY, API_URL, WORKSPACE
from tasks import NUMPY_TASKS

DATASET = "code-preference-v3"
ANNOTATORS = ["annotator2", "annotator3"]
LABELS = ["A", "B", "tie", "both_bad"]
SEED = 3

rng = random.Random(SEED)
client = rg.Argilla(api_url=API_URL, api_key=API_KEY)
dataset = client.datasets(DATASET, workspace=WORKSPACE)
user_ids = {username: client.users(username).id for username in ANNOTATORS}
records = [
    rg.Record(
        id=task["id"],
        fields={k: task[k] for k in ("prompt", "response_a", "response_b")},
        responses=[
            response
            for username, user_id in user_ids.items()
            for response in (
                rg.Response("preference", rng.choice(LABELS), user_id=user_id, status="submitted"),
                rg.Response("rationale", f"[random] placeholder vote for {username}", user_id=user_id, status="submitted"),
                rg.Response("confidence", "sure", user_id=user_id, status="submitted"),
            )
        ],
    )
    for task in NUMPY_TASKS
]
dataset.records.log(records)
print(f"submitted random votes on {len(records)} records as {', '.join(ANNOTATORS)}")
