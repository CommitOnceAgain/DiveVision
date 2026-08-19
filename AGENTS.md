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

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
