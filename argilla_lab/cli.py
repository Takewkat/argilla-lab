"""One entry point for the lab: uv run lab <command> [args].

Commands are listed in pipeline order: annotate, review, gate, training.
swe-audit reads the published SWE-bench Verified annotations and needs no Argilla.
"""

import runpy
import sys

STAGES = {
    "annotate": {
        "create-dataset": ("argilla_lab.annotate.create_dataset", "create the round's dataset in Argilla, register the round"),
    },
    "review": {
        "analyze": ("argilla_lab.review.analyze", "accuracy against the answer key, Fleiss' kappa"),
        "adjudicate": ("argilla_lab.review.adjudicate", "submit the reviewer's final labels on disputed records"),
        "bias-report": ("argilla_lab.review.bias_report", "label bias, calibration and confusion, as Markdown"),
    },
    "gate": {
        "gate": ("argilla_lab.gate.quality_gate", "three-layer quality gate, moves the round to gated or gate_failed"),
        "rounds": ("argilla_lab.core.rounds", "list the rounds, or start / validate / reopen / cancel one"),
    },
    "training": {
        "export": ("argilla_lab.training.export", "DPO/SFT export of a validated round"),
        "train-dpo": ("argilla_lab.training.train_dpo", "LoRA DPO on the export, needs uv sync --group train"),
    },
    "swe-bench": {
        "swe-audit": ("argilla_lab.swe_bench.audit", "quality audit of the published SWE-bench Verified annotations"),
    },
}
COMMANDS = {name: module for commands in STAGES.values() for name, (module, _) in commands.items()}


def usage():
    lines = ["usage: uv run lab <command> [args]"]
    for stage, commands in STAGES.items():
        lines.append(f"\n{stage}")
        lines += [f"  {name:<16}{help_text}" for name, (_, help_text) in commands.items()]
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(usage())
    module = COMMANDS[sys.argv[1]]
    sys.argv = [module, *sys.argv[2:]]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
