# Project agent memory

DiveVision has two current strands of work — see `README.md` for the full picture:

1. **Experiment workflow**: testing/comparing underwater image enhancement models (U-Shape
   Transformer, CE-VAE), benchmarked via MLflow on the LSUI/UIEB datasets.
2. **Mobile app serving the tested models**: early stage. Only a minimal FastAPI endpoint exists
   today (`divevision/src/app/main.py`); no mobile client, web app, social features, or
   geolocation exist yet. Don't imply otherwise in docs or code comments.

## Repo layout

- `divevision/models/` — vendored third-party model implementations (U-Shape Transformer, CE-VAE).
- `divevision/src/models/` — thin `AbstractModel` wrappers around those implementations
  (`abstract_model.py`, `u_shape_model.py`, `cvae_model.py`).
- `divevision/src/datasets/` — `AbstractDataset` implementations for LSUI and UIEB.
- `divevision/src/metrics/` — SSIM/PSNR metrics used by the benchmark.
- `divevision/src/test.py` — the MLflow benchmark pipeline (entry point via
  `python -m divevision.src.test`).
- `divevision/src/app/main.py` — the FastAPI serving endpoint.
- `divevision/test/` — pytest suite for models and the FastAPI app.
- `divevision/notebooks/test_model.ipynb` — manual smoke test for a model.

## Running things

- Tests: `poetry run pytest`
- Benchmark: see "Running the benchmark" in `README.md` (needs `.env` + `./mlflow_server.sh`).
- FastAPI server: `poetry run fastapi dev divevision/src/app/main.py`

## Known issues

- `download_resources.sh` only fetches model weights, not the LSUI/UIEB datasets themselves —
  those must be obtained manually (see README's Installation section for expected paths).
- The CE-VAE checkpoint download in `download_resources.sh` points to a dead Google Drive link
  (404). U-Shape Transformer's weights download works.

## Supabase: two separate projects

There are two independent Supabase projects, deliberately not sharing an API surface (a prior
design sharing one project between MLflow's tracking backend and app data caused an RLS
exposure where MLflow-adjacent policies leaked access to user photos):

- The MLflow tracking backend (`mlflow_server.sh`, `SUPABASE_POSTGRES_*` in `.env_example`) -
  unrelated to the app below, don't touch it here.
- This app's own project (auth, photo storage, `photos`/`leaderboard` tables): schema lives in
  `supabase/migrations/`, config in `supabase/config.toml`. `divevision/src/app/supabase_api.py`
  wraps `supabase-py` for it; `divevision/src/app/main.py` exposes it over FastAPI. Validate
  migrations locally with `supabase start` (requires Docker) before trusting them.

`supabase_api.py` creates a fresh client per call (`get_client()`/`get_admin_client()`) rather
than sharing one module-level client, so signing in as one user can't leak that session into a
concurrent request for another. Authenticated endpoints in `main.py` expect
`Authorization: Bearer <access_token>` and `X-Refresh-Token` headers, since `supabase-py`'s
`auth.set_session` needs both to scope a client to a user's session.

Every per-user table/bucket is RLS-scoped to `auth.uid()` - never add a blanket
`USING(true)`/`WITH CHECK(true)` policy alongside a scoped one; Postgres ORs permissive
policies together, so the blanket one silently wins and defeats the scoping.

`POST /image/` persists the original and processed photo (same relative path in both buckets)
and a `photos` row as a side effect of the existing synchronous upload-and-return flow - no
async job queue. `POST /leaderboard/` is a separate, non-user-auth path gated by a shared
secret (`LEADERBOARD_SHARED_SECRET`, `X-Leaderboard-Secret` header) for a local MLflow
benchmark script to record scores; it writes via the service-role key, which never leaves the
backend. `DELETE /account/` relies on `photos.user_id`'s `ON DELETE CASCADE` FK to clean up
rows once the auth user is deleted - it only needs to explicitly remove storage objects first.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
