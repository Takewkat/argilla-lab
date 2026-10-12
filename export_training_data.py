"""Turn the submitted preferences into DPO pairs, SFT examples and a rewrite queue.

The reviewer's label wins over the annotators' majority. A both_bad record with a
human rewrite in rewrites.json becomes training data instead of a queue entry.
An annotator answer marked unsure or cannot_judge goes to the review queue
instead of the training data, until the reviewer decides it.
"""

import json
from collections import Counter
from pathlib import Path

import argilla as rg

from config import API_KEY, API_URL, PRIVATE_DIR, WORKSPACE

DATASET = "code-preference-v2"
REVIEWER = "reviewer"
HERE = Path(__file__).parent
OUT = HERE / "out" / DATASET

rewrites = json.loads((PRIVATE_DIR / "rewrites.json").read_text())
client = rg.Argilla(api_url=API_URL, api_key=API_KEY)
reviewer_id = client.users(REVIEWER).id if client.users(REVIEWER) else None
dataset = client.datasets(DATASET, workspace=WORKSPACE)


def pair(record_id, prompt, chosen, rejected):
    return {"id": record_id, "prompt": prompt, "chosen": chosen, "rejected": rejected}


def chat(record_id, prompt, answer):
    return {"id": record_id, "messages": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]}


dpo, sft, rewrite, review, log = [], [], [], [], []
for record in sorted(dataset.records(with_responses=True), key=lambda r: r.id):
    submitted = [r for r in record.responses["preference"] if r.status == "submitted"]
    final = [r.value for r in submitted if r.user_id == reviewer_id]
    unsure = {
        r.user_id for r in record.responses["confidence"]
        if r.status == "submitted" and r.value != "sure" and r.user_id != reviewer_id
    }
    votes = [r.value for r in submitted if r.user_id != reviewer_id and r.user_id not in unsure]
    if not final and unsure:
        review.append({"id": record.id, "prompt": record.fields["prompt"], "unsure_annotators": len(unsure)})
        log.append((record.id, f"{len(unsure)} unsure -> review queue"))
        continue
    if final:
        label, source = final[0], "reviewer"
    elif votes:
        label, count = Counter(votes).most_common(1)[0]
        source = "majority"
        if count <= len(votes) / 2:
            log.append((record.id, f"skipped, no majority: {votes}"))
            continue
    else:
        log.append((record.id, "skipped, not annotated"))
        continue

    prompt = record.fields["prompt"]
    a, b = record.fields["response_a"], record.fields["response_b"]
    if label in ("A", "B"):
        chosen, rejected = (a, b) if label == "A" else (b, a)
        dpo.append(pair(record.id, prompt, chosen, rejected))
        sft.append(chat(record.id, prompt, chosen))
        log.append((record.id, f"{label} ({source}) -> dpo + sft"))
    elif label == "both_bad" and record.id in rewrites:
        human = rewrites[record.id]
        dpo += [pair(record.id, prompt, human, a), pair(record.id, prompt, human, b)]
        sft.append(chat(record.id, prompt, human))
        log.append((record.id, f"both_bad ({source}) + human rewrite -> 2 dpo + sft"))
    elif label == "both_bad":
        rewrite.append({"id": record.id, "prompt": prompt, "rejected": [a, b]})
        log.append((record.id, f"both_bad ({source}) -> rewrite queue"))
    else:
        log.append((record.id, f"tie ({source}) -> dropped, no preference signal"))

OUT.mkdir(parents=True, exist_ok=True)
outputs = (("dpo.jsonl", dpo), ("sft.jsonl", sft), ("rewrite_queue.jsonl", rewrite), ("review_queue.jsonl", review))
for name, rows in outputs:
    (OUT / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
for record_id, outcome in log:
    print(f"{record_id}  {outcome}")
print(f"\n{OUT}: " + ", ".join(f"{name} {len(rows)}" for name, rows in outputs))
