"""Run Great Expectations suites on in-memory DataFrames and print what failed, row by row.

Shared by the quality gate and the SWE-bench audit.
"""

import logging

import great_expectations as gx
from great_expectations.data_context.types.base import ProgressBarsConfig

for name in ("great_expectations", "great_expectations.data_context.types.base"):
    logging.getLogger(name).setLevel(logging.WARNING)
context = gx.get_context(mode="ephemeral")
context.variables.progress_bars = ProgressBarsConfig(globally=False)
source = context.data_sources.add_pandas("frames")


def validate(suite, frame, index_columns, show=20):
    """Run one suite on one frame, print every check with up to `show` rows it failed on, return the failed checks."""
    batch_definition = source.add_dataframe_asset(suite.name).add_batch_definition_whole_dataframe("all")
    validation = context.validation_definitions.add(
        gx.ValidationDefinition(name=suite.name, data=batch_definition, suite=context.suites.add(suite))
    )
    result = validation.run(
        batch_parameters={"dataframe": frame.reset_index(drop=True)},
        result_format={"result_format": "COMPLETE", "unexpected_index_column_names": index_columns},
    )
    print(f"\n[{suite.name}] {len(frame)} rows")
    failures = []
    for check in result.results:
        print(f"  {'PASS' if check.success else 'FAIL'}  {check.expectation_config.description}")
        if not check.success:
            unexpected = check.result.get("unexpected_index_list") or []
            names = sorted({"/".join(str(row[column]) for column in index_columns) for row in unexpected})
            more = f", ... {len(names) - show} more" if len(names) > show else ""
            print(f"        {len(names)}: {', '.join(names[:show])}{more}")
            failures.append(f"{suite.name}: {check.expectation_config.description} ({len(names) or 'table'})")
    return failures
