"""The rubrics the lab knows, and which one a dataset uses."""

from argilla_lab.rubrics import code_preference, swe_verified

RUBRICS = {rubric.name: rubric for rubric in (code_preference.RUBRIC, swe_verified.RUBRIC)}


def for_dataset(name):
    """The rubric of a dataset, from its name prefix: code-preference-v3 uses code-preference."""
    for prefix, rubric in RUBRICS.items():
        if name == prefix or name.startswith(f"{prefix}-"):
            return rubric
    raise SystemExit(f"{name}: no rubric, dataset names start with one of {', '.join(RUBRICS)}")
