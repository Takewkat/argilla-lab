"""A rubric as data: what annotators answer, how each answer is checked, how votes become a decision.

The checks (core/checks.py) and the report (review/report.py) read a rubric and nothing else,
so one gate and one report serve every rubric in argilla_lab/rubrics/.
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    """One scored question. Its answers are aggregated into the record's decision."""

    name: str
    labels: tuple
    ordinal: bool = False  # labels on a scale: distances count, and "max" aggregation applies
    drop_at: float | None = None  # ordinal: a record whose aggregated label reaches this is dropped
    notes: str | None = None  # column of the free-text explanation for this question
    gold: bool = False  # answer_key.json holds a gold label for it

    @property
    def scaled(self):
        """An ordinal question with more than two labels, where a distance to peers means something."""
        return self.ordinal and len(self.labels) > 2


@dataclass(frozen=True)
class Rubric:
    name: str
    questions: tuple[Question, ...]
    aggregation: str  # "majority": the most voted label wins; "max": the most severe label wins
    min_votes: int  # counted votes a record needs
    min_notes: int  # characters an explanation needs
    confidence: str | None = None  # column of the annotator's confidence
    confidence_labels: tuple = ()
    sure: tuple = ()  # confidence labels whose answers count; empty: every answer counts
    fields: tuple = ()  # other answer fields to load, read by prepare() and answer_checks
    prepare: Callable | None = None  # adds the derived columns the answer_checks read
    answer_checks: tuple = ()  # Great Expectations checks specific to this rubric
    min_annotators: int = 2
    min_profile: int = 1  # answers before an annotator is profiled
    min_sure: int = 3  # sure answers before accuracy on the gold key is measured
    min_gold_accuracy: float = 0.6
    max_deferred: float = 0.5
    max_peer_bias: float = 0.75

    def __post_init__(self):
        if self.aggregation not in ("majority", "max"):
            raise ValueError(f"{self.name}: aggregation is majority or max, not {self.aggregation}")

    @property
    def gold_question(self):
        return next((q for q in self.questions if q.gold), None)

    @property
    def answer_columns(self):
        """Every column an annotator fills in, in a stable order."""
        columns = [q.name for q in self.questions] + [q.notes for q in self.questions if q.notes]
        columns += [self.confidence] if self.confidence else []
        return list(dict.fromkeys(columns + list(self.fields)))
