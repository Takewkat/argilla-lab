"""Compare the submitted preferences across annotators and against the answer key."""

import json
from collections import Counter

import argilla as rg

from config import API_KEY, API_URL, PRIVATE_DIR, WORKSPACE

DATASET = "code-preference-v3"
LABELS = ["A", "B", "tie", "both_bad"]
REVIEWER = "reviewer"

key = json.loads((PRIVATE_DIR / "answer_key.json").read_text())
client = rg.Argilla(api_url=API_URL, api_key=API_KEY)
usernames = {u.id: u.username for u in client.users}
dataset = client.datasets(DATASET, workspace=WORKSPACE)

rows = []
for record in dataset.records(with_responses=True):
    submitted = [r for r in record.responses["preference"] if r.status == "submitted"]
    rationales = {r.user_id: r.value for r in record.responses["rationale"] if r.status == "submitted"}
    votes = {usernames[r.user_id]: r.value for r in submitted}
    why = {usernames[uid]: text for uid, text in rationales.items()}
    doubt = {usernames[r.user_id]: r.value for r in record.responses["confidence"] if r.status == "submitted" and r.value != "sure"}
    final = votes.pop(REVIEWER, None)
    rows.append((record.id, votes, why, final, doubt))

rows.sort()
hits = Counter()
seen = Counter()
deferred = Counter()
for record_id, votes, why, final, doubt in rows:
    expected = key[record_id]["preference"]
    if len(votes) < 2:
        marker = "single"
    else:
        marker = "agree" if len(set(votes.values())) == 1 else "SPLIT"
    print(f"\n{record_id} [{marker}] key={expected}: {key[record_id]['why']}")
    for user, vote in sorted(votes.items()):
        if user in doubt:
            deferred[user] += 1
            print(f"  ?? {user:<12} {vote:<9} [{doubt[user]}] {why.get(user, '')}")
            continue
        seen[user] += 1
        hits[user] += vote == expected
        flag = "ok " if vote == expected else "-- "
        print(f"  {flag}{user:<12} {vote:<9} {why.get(user, '')}")
    if final:
        print(f"  => {REVIEWER:<12} {final:<9} {why.get(REVIEWER, '')}")

print("\nAccuracy against the key (sure answers only), and answers deferred to the reviewer:")
for user in sorted(set(seen) | set(deferred)):
    print(f"  {user:<12} {hits[user]}/{seen[user]} sure, {deferred[user]} deferred")

# Fleiss' kappa over the records every annotator submitted
full = [votes for _, votes, _, _, _ in rows if len(votes) == len(seen) and len(seen) > 1]
if full:
    n = len(seen)
    counts = [[list(v.values()).count(label) for label in LABELS] for v in full]
    p_bar = sum((sum(c * c for c in row) - n) / (n * (n - 1)) for row in counts) / len(counts)
    p_j = [sum(row[j] for row in counts) / (len(counts) * n) for j in range(len(LABELS))]
    p_e = sum(p * p for p in p_j)
    kappa = (p_bar - p_e) / (1 - p_e) if p_e < 1 else 1.0
    print(f"\nFleiss' kappa over {len(full)} fully annotated records: {kappa:.2f}")
