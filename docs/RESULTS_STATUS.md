# Results Status

This file is for tracking local reproduction progress before public release. Generated logs and artifacts are not committed.

Known local run directories:

```text
backend_logs/20260516_044034/
backend_logs/android_rerun_20260518_002340/
```

The main backend run completed training and classification for:

```text
iscx-vpn
iscx-nonvpn
iscx-tor
iscx-nontor
ios
android
```

The ISCX instance-level and class-level explanation jobs completed for the default methods in `scripts/run_backend_all.sh`.

The cross-platform localization jobs are heavier. Treat IOS and Android explanation outputs as local run artifacts and verify the latest `backend_logs/*/manifest.tsv` before reporting final public numbers.
