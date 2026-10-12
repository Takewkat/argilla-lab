"""Create the code-preference dataset and three annotator accounts in a local Argilla.

Round 2 (code-preference-v2): guidelines updated after adjudicating round 1,
plus a confidence question so an unsure annotator routes the record to a reviewer.

Round 3 (code-preference-v3): same guidelines, the five numpy and pandas pairs only.
"""

import argilla as rg

import rounds
from config import API_KEY, API_URL, WORKSPACE, env
from tasks import NUMPY_TASKS

DATASET = "code-preference-v3"
ANNOTATORS = ["iana", "annotator2", "annotator3"]
ANNOTATOR_PASSWORD = env("ANNOTATOR_PASSWORD")

GUIDELINES = """\
You see a coding prompt and two answers, A and B.

1. Pick the better answer. Correctness first, then following the prompt's
   instructions, then security, then readability.
2. `tie`: both are acceptable and differ only in style.
3. `both_bad`: neither answer does what the prompt asks.
4. Tag every issue you found, in either answer.
5. Explain your choice in one or two sentences. Name the line or the behaviour.
6. Say how sure you are. `unsure` and `cannot_judge` are not failures: the
   record goes to a reviewer instead of into the training data.

Do not run the code. Judge it the way a reviewer would.

Lessons from round 1:
- Reread the prompt for explicit requirements (order, error type, length)
  and check each answer against every one of them.
- A longer or more detailed answer is not better when the prompt limits it.
- If you spotted a problem but cannot tell how bad it is, pick `unsure`.
"""

client = rg.Argilla(api_url=API_URL, api_key=API_KEY)
workspace = client.workspaces(WORKSPACE)

for username in ANNOTATORS:
    user = client.users(username)
    if user is None:
        user = rg.User(username=username, password=ANNOTATOR_PASSWORD, role="annotator", client=client).create()
    if username not in [u.username for u in workspace.users]:
        workspace.add_user(user)

settings = rg.Settings(
    guidelines=GUIDELINES,
    fields=[
        rg.TextField(name="prompt", title="Prompt", use_markdown=True),
        rg.TextField(name="response_a", title="Answer A", use_markdown=True),
        rg.TextField(name="response_b", title="Answer B", use_markdown=True),
    ],
    questions=[
        rg.LabelQuestion(
            name="preference",
            title="Which answer is better?",
            labels=["A", "B", "tie", "both_bad"],
        ),
        rg.MultiLabelQuestion(
            name="issues",
            title="Issues found (in either answer)",
            labels=["bug", "security", "ignores_instructions", "performance", "style_only", "none"],
            required=False,
        ),
        rg.TextQuestion(name="rationale", title="Why?", use_markdown=False),
        rg.LabelQuestion(
            name="confidence",
            title="How sure are you?",
            labels=["sure", "unsure", "cannot_judge"],
        ),
    ],
    distribution=rg.TaskDistribution(min_submitted=1),
)

if client.datasets(DATASET, workspace=WORKSPACE) is not None:
    raise SystemExit(f"{DATASET} already exists, delete it in the UI first to start over")

dataset = rg.Dataset(name=DATASET, workspace=WORKSPACE, settings=settings, client=client)
dataset.create()
dataset.records.log(
    [
        rg.Record(
            id=task["id"],
            fields={k: task[k] for k in ("prompt", "response_a", "response_b")},
        )
        for task in NUMPY_TASKS
    ]
)

print(f"{DATASET}: {len(NUMPY_TASKS)} records")
rounds.move(DATASET, "annotating", "round created")
print(f"UI: {API_URL}, annotators {', '.join(ANNOTATORS)}, password in .env")
