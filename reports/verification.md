# Software verification

Verified on 22 October 2026 in a newly created virtual environment using
Python 3.12.14 and CPU PyTorch 2.6.0.

- Installed the project and declared test dependencies; `python -m pip check` passed.
- Built and installed the distribution wheel.
- Ran the installed package from `/tmp` using isolated Python (`python -I`) and
  pytest's importlib mode: **32 tests passed**.
- Flake8's syntax/undefined-name checks passed (`E9,F63,F7,F82`).
- Initialized DVC and successfully checked `dvc dag --md` and `dvc stage list`
  without fetching data or running training.
- Checked the committed provisional CSVs for disjoint filenames, valid labels,
  deterministic allocation and checksum consistency: 14,701 / 3,151 / 3,151 rows.
- Recalculated archived per-attribute accuracy and macro F1 from existing
  confusion matrices, checked their counts and accuracy against the original JSON,
  and rendered those matrices. The source counts were not independently rerun.

The regression suite generates tiny random images to test the complete software
path (audit → partitions → preprocessing → model selection → final report),
identity/duplicate grouping, data mutation detection, missing/ambiguous images,
best-epoch metric alignment, validation-only Hyperopt and single test access,
including a reporting failure after test use. These fixtures are not research data.

Hyperopt emits upstream deprecation warnings concerning pkg_resources and
UTC datetime construction. They do not fail the tests.

No training or testing on the original face dataset was performed: its images
are unavailable. No independent model-performance claim follows from these checks.
Python 3.10 is configured in CI but was not executed in this local environment.
