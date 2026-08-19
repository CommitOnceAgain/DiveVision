# MLflow tracking backend runs on Supabase Postgres + S3-compatible storage

`mlflow_server.sh` points MLflow's backend store at a Supabase-hosted Postgres database and its
artifact store at S3-compatible storage (`.env_example`'s `SUPABASE_POSTGRES_*` and `AWS_*`
variables), instead of MLflow's default local file store. This is a deliberate choice to run
Benchmark Runs against shared, durable infrastructure from the start rather than local SQLite/disk
storage, and it reuses the same Supabase project the app strand is expected to depend on for
storage and auth later (see issue #4), rather than standing up separate infrastructure for
experiment tracking alone.

Reversing this later means migrating existing MLflow run history, not just a config change.
