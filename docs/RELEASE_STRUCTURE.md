# Release Structure

The current repository keeps the executable research scripts at the root so existing reproduction commands continue to work.

Recommended public layout:

```text
Traffic-Explainer/
  README.md
  requirements.txt
  config.py
  dataset.py
  model.py
  utils.py
  main.py
  test.py
  explain.py
  explain_class.py
  run_classification.sh
  run_classification_anomaly.sh
  run_explanations.sh
  run_explanations_anomaly.sh
  run_explanations_class.sh
  scripts/
    repro_integrity_check.py
    run_backend_all.sh
    summarize_backend_results.py
  docs/
    DATA.md
    DATA_REPRODUCTION.md
    PUBLIC_RELEASE_CHECKLIST.md
    REPRODUCTION.md
    RELEASE_STRUCTURE.md
    RESULTS_STATUS.md
    SPLITCAP.md
```

Ignored local paths:

```text
dataset/
dataset_localization/
model/
res/
backend_logs/
checkpoints/
```

For a later package-style release, the code can be moved into `src/e4traffic/` and the command-line entrypoints can become console scripts. That refactor should happen after long-running reproduction jobs finish, because it changes import paths and process assumptions.
