"""A Markdown quality report from an evaluation, its sections chosen by the rubric.

Every rubric gets decisions, agreement and annotators. A gold question adds label bias,
calibration against correctness and a confusion matrix. A scaled question adds the
distance to peers, and max aggregation adds what the aggregation rule changes.
"""

import numpy as np

from argilla_lab.core.agreement import interval, krippendorff_alpha, nominal
from argilla_lab.core.checks import counting_rule, peer_distance

SMALL_N = 5
TOP = 10


def md_table(headers, rows):
    """A Markdown table from a header list and a list of row lists."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def n_note(n):
    return f"{n} (small n)" if n < SMALL_N else str(n)


def codes(question, values):
    """Labels as numbers in the rubric's order, so agreement works on any label type."""
    return values.map({label: i for i, label in enumerate(question.labels)}).astype(float)


def alpha(units, distance):
    """Krippendorff's alpha, or None when no record has two answers to compare."""
    if not any(np.count_nonzero(~np.isnan(unit)) >= 2 for unit in units):
        return None
    return krippendorff_alpha(units, distance)


def decisions(rubric, evaluation):
    records = evaluation.records
    lines = ["## Decisions", "", f"Counted votes: {counting_rule(rubric)}.", ""]
    if rubric.aggregation == "majority":
        for q in rubric.questions:
            decided = (records[f"{q.name}_votes"] >= rubric.min_votes) & (records[f"{q.name}_share"] > 0.5)
            lines.append(
                f"- {q.name}: {int(decided.sum())} of {len(records)} records have {rubric.min_votes}+ counted votes "
                "and a strict majority"
            )
        return lines
    flipped = records[~records["same_decision"]]
    return lines + [
        f"- dropped when the most severe label wins: {int(records['dropped'].sum())} ({records['dropped'].mean():.1%})",
        f"- dropped on the median label: {int(records['dropped_median'].sum())} "
        f"({records['dropped_median'].mean():.1%})",
        f"- decision depends on the rule: {len(flipped)} records, {int((flipped['flags'] == 1).sum())} dropped on a "
        f"single annotator's flag, {int((flipped['flags'] > 1).sum())} on flags for different questions",
    ]


def agreement(rubric, evaluation):
    rows = []
    for q in rubric.questions:
        units = [codes(q, group).to_numpy() for _, group in evaluation.answers.groupby("record_id")[q.name]]
        values = [alpha(units, nominal), alpha(units, interval) if q.ordinal else None]
        rows.append([q.name, *("-" if value is None else f"{value:.2f}" for value in values)])
    return [
        "## Agreement, Krippendorff's alpha",
        "",
        md_table(["question", "nominal", "interval"], rows),
        "",
        "Nominal: any two labels differ as much. Interval: on a scale, 0 vs 3 is further apart than 1 vs 2.",
    ]


def annotators(rubric, evaluation):
    profiles = evaluation.annotators
    columns = {"annotator": "annotator", "answers": "answers"}
    if rubric.sure:
        columns |= {"sure": "sure", "deferred_rate": "deferred"}
    if rubric.gold_question:
        columns["gold_accuracy"] = "gold accuracy"
    columns |= {f"{q.name}_vs_peers": f"{q.name} vs peers" for q in rubric.questions if q.scaled}
    if rubric.aggregation == "max":
        columns["sole_drops"] = "sole drops"
        profiles, order = profiles.sort_values("sole_drops", ascending=False), "most sole drops first"
    elif rubric.gold_question:
        profiles = profiles.sort_values("gold_accuracy", na_position="first")
        order = "no sure answer, then lowest gold accuracy first"
    else:
        order = "by name"
    shown = profiles.head(TOP)

    def cell(column, value):
        if column == "annotator":
            return value
        if value != value:  # NaN: nothing to measure, e.g. no sure answer
            return "-"
        if column in ("deferred_rate", "gold_accuracy"):
            return f"{value:.0%}"
        if column.endswith("_vs_peers"):
            return f"{value:+.2f}"
        return str(int(value))

    rows = [[cell(column, row[column]) for column in columns] for _, row in shown.iterrows()]
    return [
        "## Annotators",
        "",
        f"{len(shown)} of {len(profiles)} shown, {order}. `vs peers`: mean distance to the others on the same record.",
        "",
        md_table(list(columns.values()), rows),
    ]


def label_bias(rubric, evaluation):
    """Share of each label picked minus its share in the gold, per annotator, in points."""
    q = rubric.gold_question
    if q is None:
        return []
    rows = []
    for annotator, answers in evaluation.answers.groupby("annotator"):
        picked = answers[q.name].value_counts(normalize=True).reindex(q.labels, fill_value=0)
        expected = answers[f"{q.name}_gold"].value_counts(normalize=True).reindex(q.labels, fill_value=0)
        rows.append([annotator, n_note(len(answers)), *(f"{value:+.0f}" for value in (picked - expected) * 100)])
    return [
        "## Label bias",
        "",
        "Points picked minus points in the gold. `A` well above `B` with a balanced gold is position bias; "
        "`both_bad` below zero means real defects get waved through.",
        "",
        md_table(["annotator", "n", *q.labels], rows),
    ]


def calibration(rubric, evaluation):
    """Confidence against correctness with a gold key, against the distance to peers without one."""
    if not rubric.confidence:
        return []
    answers = evaluation.answers
    level = answers[rubric.confidence].astype(object).where(answers[rubric.confidence].notna(), "not asked")
    order = [*rubric.confidence_labels, "not asked"]
    gold = rubric.gold_question
    if gold:
        right = answers[gold.name] == answers[f"{gold.name}_gold"]
        sure = level.isin(rubric.sure)
        deferred = ~sure & (level != "not asked")
        headers = ["annotator", "n", "sure right", "sure wrong", "deferred right", "deferred wrong", "not asked"]
        note = "`sure wrong` reaches the training data unreviewed: the costly cell. `deferred right` costs reviewer time."
        rows = []
        for annotator in [*sorted(answers["annotator"].unique()), "all"]:
            mine = answers["annotator"] == annotator if annotator != "all" else answers["annotator"].notna()
            rows.append([
                annotator,
                n_note(int(mine.sum())),
                *(int((mine & cell).sum()) for cell in (sure & right, sure & ~right, deferred & right, deferred & ~right)),
                int((mine & (level == "not asked")).sum()),
            ])
    else:
        scaled = [q for q in rubric.questions if q.scaled]
        distance = {q.name: peer_distance(answers, q.name).abs() for q in scaled}
        headers = ["confidence", "n", *(f"{q.name} distance to peers" for q in scaled)]
        note = "A calibrated annotator is closer to peers when more confident."
        rows = [
            [label, n_note(int(mask.sum())), *(f"{distance[q.name][mask].mean():.2f}" for q in scaled)]
            for label in order
            if (mask := level == label).any()
        ]
    return ["## Calibration", "", note, "", md_table(headers, rows)]


def confusion(rubric, evaluation):
    q = rubric.gold_question
    if q is None:
        return []
    answers = evaluation.answers
    rows = []
    for label in q.labels:
        given = answers.loc[answers[f"{q.name}_gold"] == label, q.name]
        rows.append([f"gold {label}", n_note(len(given)), *(int((given == other).sum()) for other in q.labels)])
    return [
        "## Confusion",
        "",
        "Off-diagonal cells are the systematic mistakes. A whole row off the diagonal points at a guideline gap.",
        "",
        md_table(["gold", "n", *(f"answered {label}" for label in q.labels)], rows),
    ]


def write(rubric, evaluation, title, path, failed=(), notes=()):
    """Build the report, write it to path, return its lines."""
    answers = evaluation.answers
    lines = [
        f"# {title}",
        "",
        f"{len(answers)} answers on {len(evaluation.records)} records by {answers['annotator'].nunique()} annotators, "
        f"{rubric.name} rubric. Small n below {SMALL_N}.",
        *(f"\n{note}" for note in notes),
    ]
    for section in (decisions, agreement, annotators, label_bias, calibration, confusion):
        part = section(rubric, evaluation)
        lines += ["", *part] if part else []
    if failed:
        lines += ["", f"## Checks failed: {len(failed)}", "", *(f"- {name}" for name in failed)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return lines
