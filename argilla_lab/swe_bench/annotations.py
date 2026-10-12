"""The human annotations OpenAI published with SWE-bench Verified, kept in data/ as published.

Source: https://cdn.openai.com/introducing-swe-bench-verified/swe-bench-annotation-results.zip
(Last-Modified 2024-08-12), from https://openai.com/index/introducing-swe-bench-verified/

Two tables, read straight from the archive:
- per_annotator: one row per sample and annotator, 3 annotators per sample
- ensembled: one row per sample, the label kept and whose label it was

answers() reshapes the first for core/checks.py, under the swe-verified rubric.

The file is checked against the sha256 of the published archive, so an edited or
replaced copy fails to load instead of silently changing every metric.
"""

import hashlib
import io
import zipfile

import pandas as pd

from argilla_lab.core.paths import DATA_DIR

ARCHIVE = DATA_DIR / "swe-bench-annotation-results.zip"
SHA256 = "b9297c31f6c2ef712218b5bb2830bd47a7f600dedf6b927191d0170177266ab9"
SEVERITIES = ["underspecified", "false_negative", "other_major_issues"]


def archive():
    """The archive bytes, checked against the sha256 of the published file."""
    data = ARCHIVE.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"{ARCHIVE} has sha256 {digest}, expected {SHA256}: not the published file")
    return data


def load():
    """(per_annotator, ensembled) DataFrames, severity labels as numbers."""
    with zipfile.ZipFile(io.BytesIO(archive())) as zf:
        per_annotator = pd.read_csv(zf.open("samples_with_3_annotations_public.csv"))
        ensembled = pd.read_csv(zf.open("ensembled_annotations_public.csv"))
    for df in (per_annotator, ensembled):
        df[SEVERITIES] = df[SEVERITIES].apply(pd.to_numeric, errors="coerce")
    per_annotator["user_id"] = per_annotator["user_id"].astype(str)
    return per_annotator, ensembled


def answers(per_annotator):
    """The per-annotator table shaped for core/checks.py: record_id, annotator, then the rubric's columns."""
    frame = per_annotator.rename(columns={"instance_id": "record_id", "user_id": "annotator"})
    return frame, sorted(frame["record_id"].unique())
