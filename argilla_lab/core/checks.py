"""The three layers of checks, built from a rubric alone: answers, annotators, records.

evaluate() derives the annotator and record frames, validate() runs one Great Expectations
suite per layer on them. The answers frame has one row per record and annotator:
record_id, annotator, the rubric's answer_columns, and <question>_gold for a gold question.
"""

import copy
from dataclasses import dataclass

import great_expectations as gx
import numpy as np
import pandas as pd

from argilla_lab.core.gx_runner import validate as run_suite

PLACEHOLDERS = [r"^\s*$", r"(?i)^\s*(test|n/?a|none|todo|asdf|-+|\.+)\s*$"]


@dataclass
class Evaluation:
    answers: pd.DataFrame
    annotators: pd.DataFrame
    records: pd.DataFrame
    trusted: list


def counting_rule(rubric):
    """Which votes count toward a record, in words."""
    who = "trusted annotators (gold key)" if rubric.gold_question else "every annotator (no gold key)"
    what = f"{'/'.join(rubric.sure)} answers" if rubric.sure else "every answer"
    return f"{what} from {who}"


def counted(rubric, answers):
    """The answers that count: sure ones only, when the rubric has a notion of sure."""
    if not rubric.sure:
        return pd.Series(True, index=answers.index)
    return answers[rubric.confidence].isin(rubric.sure)


def peer_distance(answers, column):
    """Each answer minus the mean of the other annotators on the same record."""
    group = answers.groupby("record_id")[column]
    peers = (group.transform("sum") - answers[column]) / (group.transform("count") - 1)
    return answers[column] - peers


def flags(rubric, answers):
    """Whether each answer alone reaches a drop threshold, so would drop its record under max aggregation."""
    flagged = pd.Series(False, index=answers.index)
    for q in rubric.questions:
        if q.drop_at is not None:
            flagged |= answers[q.name] >= q.drop_at
    return flagged


def annotator_profiles(rubric, answers):
    is_counted = counted(rubric, answers)
    by_annotator = answers["annotator"]
    profiles = answers.groupby("annotator").size().rename("answers").to_frame()
    profiles["sure"] = is_counted.groupby(by_annotator).sum()
    profiles["deferred_rate"] = 1 - profiles["sure"] / profiles["answers"]
    gold = rubric.gold_question
    if gold:
        sure = answers[is_counted]
        correct = sure[gold.name] == sure[f"{gold.name}_gold"]
        profiles["gold_accuracy"] = correct.groupby(sure["annotator"]).mean()
    for q in rubric.questions:
        if q.scaled:
            profiles[f"{q.name}_vs_peers"] = peer_distance(answers, q.name).groupby(by_annotator).mean()
    if rubric.aggregation == "max":
        flagged = flags(rubric, answers) & is_counted
        alone = flagged & (flagged.groupby(answers["record_id"]).transform("sum") == 1)
        profiles["sole_drops"] = alone.groupby(by_annotator).sum()
    return profiles.reset_index()


def trusted_annotators(rubric, profiles):
    """Annotators whose votes count. Without a gold key there is nothing to exclude anyone on."""
    if rubric.gold_question is None:
        return list(profiles["annotator"])
    ok = (profiles["sure"] >= rubric.min_sure) & (profiles["gold_accuracy"] >= rubric.min_gold_accuracy)
    return list(profiles.loc[ok, "annotator"])


def record_decisions(rubric, answers, record_ids, trusted):
    """One row per record of the dataset, answered or not: counted votes and the aggregated label."""
    votes = answers[counted(rubric, answers) & answers["annotator"].isin(trusted)]
    records = pd.DataFrame({"record_id": record_ids})
    for q in rubric.questions:
        grouped = votes.groupby("record_id")[q.name]
        stats = pd.DataFrame({f"{q.name}_votes": grouped.count()})
        if rubric.aggregation == "majority":
            stats[f"{q.name}_label"] = grouped.agg(lambda s: s.mode().iloc[0] if s.notna().any() else None)
            stats[f"{q.name}_share"] = grouped.agg(
                lambda s: s.value_counts(normalize=True).iloc[0] if s.notna().any() else np.nan
            )
        else:
            stats[f"{q.name}_label"] = grouped.max()
            stats[f"{q.name}_median"] = grouped.median()
        records = records.merge(stats, left_on="record_id", right_index=True, how="left")
        records[f"{q.name}_votes"] = records[f"{q.name}_votes"].fillna(0)
    if rubric.aggregation == "max":
        droppable = [q for q in rubric.questions if q.drop_at is not None]
        records["dropped"] = np.logical_or.reduce([records[f"{q.name}_label"] >= q.drop_at for q in droppable])
        records["dropped_median"] = np.logical_or.reduce([records[f"{q.name}_median"] >= q.drop_at for q in droppable])
        records["same_decision"] = records["dropped"] == records["dropped_median"]
        per_record = flags(rubric, votes).groupby(votes["record_id"]).sum()
        records["flags"] = records["record_id"].map(per_record).fillna(0).astype(int)
    return records


def evaluate(rubric, answers, record_ids):
    if rubric.prepare:
        answers = rubric.prepare(answers)
    profiles = annotator_profiles(rubric, answers)
    trusted = trusted_annotators(rubric, profiles)
    return Evaluation(answers, profiles, record_decisions(rubric, answers, record_ids, trusted), trusted)


def answer_suite(rubric):
    expect = gx.expectations
    checks = [
        # Row count first: the other checks pass on an empty table
        expect.ExpectTableRowCountToBeBetween(min_value=1, description="At least one submitted answer"),
        expect.ExpectCompoundColumnsToBeUnique(
            column_list=["record_id", "annotator"], description="One answer per annotator and record"
        ),
    ]
    # Not-null next to each in-set check, which skips nulls
    for q in rubric.questions:
        labels = ", ".join(map(str, q.labels))
        checks += [
            expect.ExpectColumnValuesToNotBeNull(column=q.name, description=f"{q.name}: answered"),
            expect.ExpectColumnValuesToBeInSet(
                column=q.name, value_set=list(q.labels), description=f"{q.name}: one of {labels}"
            ),
        ]
        if q.gold:
            checks.append(
                expect.ExpectColumnValuesToNotBeNull(
                    column=f"{q.name}_gold", description=f"{q.name}: a gold label in answer_key.json"
                )
            )
    if rubric.confidence:
        labels = ", ".join(map(str, rubric.confidence_labels))
        checks += [
            expect.ExpectColumnValuesToNotBeNull(column=rubric.confidence, description="confidence: given"),
            expect.ExpectColumnValuesToBeInSet(
                column=rubric.confidence,
                value_set=list(rubric.confidence_labels),
                description=f"confidence: one of {labels}",
            ),
        ]
    for notes in [q.notes for q in rubric.questions if q.notes]:
        checks += [
            expect.ExpectColumnValuesToNotBeNull(column=notes, description=f"{notes}: present"),
            expect.ExpectColumnValuesToNotMatchRegexList(
                column=notes, regex_list=PLACEHOLDERS, description=f"{notes}: not a placeholder"
            ),
            expect.ExpectColumnValueLengthsToBeBetween(
                column=notes, min_value=rubric.min_notes, description=f"{notes}: {rubric.min_notes}+ chars"
            ),
        ]
    # Copies of the rubric's own checks
    checks += [copy.deepcopy(check) for check in rubric.answer_checks]
    return gx.ExpectationSuite(name=f"{rubric.name}_answers", expectations=checks)


def annotator_suite(rubric):
    expect = gx.expectations
    checks = [
        expect.ExpectTableRowCountToBeBetween(
            min_value=rubric.min_annotators,
            description=f"{rubric.min_annotators}+ annotators, agreement needs a second opinion",
        ),
    ]
    if rubric.min_profile > 1:
        checks.append(
            expect.ExpectColumnValuesToBeBetween(
                column="answers", min_value=rubric.min_profile, description=f"{rubric.min_profile}+ answers to profile"
            )
        )
    # Each ratio next to the count it divides by
    if rubric.gold_question:
        checks += [
            expect.ExpectColumnValuesToBeBetween(
                column="sure",
                min_value=rubric.min_sure,
                description=f"Enough sure answers to measure accuracy ({rubric.min_sure}+)",
            ),
            expect.ExpectColumnValuesToBeBetween(
                column="gold_accuracy",
                min_value=rubric.min_gold_accuracy,
                description=f"Sure answers match the gold key ({rubric.min_gold_accuracy:.0%}+)",
            ),
        ]
    if rubric.sure:
        checks.append(
            expect.ExpectColumnValuesToBeBetween(
                column="deferred_rate",
                max_value=rubric.max_deferred,
                description=f"At most {rubric.max_deferred:.0%} of answers deferred to the reviewer",
            )
        )
    for q in rubric.questions:
        if q.scaled:
            checks.append(
                expect.ExpectColumnValuesToBeBetween(
                    column=f"{q.name}_vs_peers",
                    min_value=-rubric.max_peer_bias,
                    max_value=rubric.max_peer_bias,
                    description=f"{q.name}: within {rubric.max_peer_bias} points of peers on the same records",
                )
            )
    return gx.ExpectationSuite(name=f"{rubric.name}_annotators", expectations=checks)


def record_suite(rubric):
    expect = gx.expectations
    checks = [expect.ExpectTableRowCountToBeBetween(min_value=1, description="The dataset has records")]
    for q in rubric.questions:
        checks.append(
            expect.ExpectColumnValuesToBeBetween(
                column=f"{q.name}_votes",
                min_value=rubric.min_votes,
                description=f"{q.name}: {rubric.min_votes}+ counted votes",
            )
        )
        if rubric.aggregation == "majority":
            checks.append(
                expect.ExpectColumnValuesToBeBetween(
                    column=f"{q.name}_share",
                    min_value=0.5,
                    strict_min=True,
                    description=f"{q.name}: a strict majority among the counted votes",
                )
            )
    if rubric.aggregation == "max":
        checks.append(
            expect.ExpectColumnValuesToBeInSet(
                column="same_decision",
                value_set=[True],
                description="Keep/drop does not depend on the rule (max severity vs median)",
            )
        )
    return gx.ExpectationSuite(name=f"{rubric.name}_records", expectations=checks)


def validate(rubric, evaluation, show=20):
    """Run the three suites, print every check, return the failed ones."""
    return [
        *run_suite(answer_suite(rubric), evaluation.answers, ["record_id", "annotator"], show),
        *run_suite(annotator_suite(rubric), evaluation.annotators, ["annotator"], show),
        *run_suite(record_suite(rubric), evaluation.records, ["record_id"], show),
    ]
