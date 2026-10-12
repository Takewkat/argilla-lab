"""Status of each annotation round, kept with the reviewer material in rounds.json.

    annotating -> gated or gate_failed   quality_gate.py
    gated      -> validated              a reviewer, with a reason
    validated  -> exported               export_training_data.py
    validated  -> annotating             a reviewer reopens it, the answers must change
    any round not exported -> canceled

Every move is appended to the round's history: what, when, who and why.

Usage:
    python rounds.py                           list the rounds
    python rounds.py start <dataset>
    python rounds.py validate <dataset> <reason>
    python rounds.py reopen <dataset> <reason>
    python rounds.py cancel <dataset> <reason>
"""

import getpass
import json
import sys
from datetime import datetime, timezone

from config import PRIVATE_DIR

ROUNDS = PRIVATE_DIR / "rounds.json"
# Status -> the statuses it may move to. None is a round not registered yet.
TRANSITIONS = {
    None: {"annotating"},
    "annotating": {"gated", "gate_failed", "canceled"},
    "gate_failed": {"gated", "gate_failed", "canceled"},
    "gated": {"gated", "gate_failed", "validated", "canceled"},
    "validated": {"exported", "annotating", "canceled"},
    "exported": {"exported"},
    "canceled": set(),
}
COMMANDS = {"validate": "validated", "reopen": "annotating", "cancel": "canceled"}


def load():
    return json.loads(ROUNDS.read_text()) if ROUNDS.exists() else {}


def get(dataset):
    """The round's record: status, history and the facts stored along the way. Empty when unregistered."""
    return load().get(dataset, {"status": None, "history": []})


def move(dataset, new_status, reason, **facts):
    """Move a round to new_status and log it, or exit 1 when the move is not allowed."""
    rounds = load()
    round_ = rounds.setdefault(dataset, {"status": None, "history": []})
    current = round_["status"]
    if new_status not in TRANSITIONS[current]:
        raise SystemExit(f"{dataset}: cannot go from {current or 'unregistered'} to {new_status}")
    round_["status"] = new_status
    round_.update(facts)
    round_["history"].append(
        {
            "status": new_status,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "by": getpass.getuser(),
            "reason": reason,
        }
    )
    ROUNDS.write_text(json.dumps(rounds, indent=2) + "\n")
    print(f"{dataset}: {current or 'unregistered'} -> {new_status} ({reason})")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        for name, round_ in sorted(load().items()):
            last = round_["history"][-1]
            print(f"{name:<22} {round_['status']:<12} {last['at']}  {last['reason']}")
    elif args[0] == "start" and len(args) == 2:
        move(args[1], "annotating", "round created")
    elif args[0] in COMMANDS and len(args) >= 3:
        move(args[1], COMMANDS[args[0]], " ".join(args[2:]))
    else:
        raise SystemExit(__doc__)
