# DiveVision

DiveVision explores solutions for **underwater image restoration / enhancement**. The project
has two current strands of work:

1. **Experiment workflow** — testing and comparing underwater image enhancement models
   (currently [U-Shape Transformer](https://github.com/LintaoPeng/U-shape_Transformer_for_Underwater_Image_Enhancement)
   and [CE-VAE](https://github.com/iN1k1/ce-vae-underwater-image-enhancement)), benchmarked with
   MLflow against the LSUI and UIEB datasets.
2. **Mobile app serving the tested models** — very early stage. The current goal is simply to
   serve a model's output to a mobile client. Social-network features and photo geolocation are
   **future work**, not part of the current scope.

## What exists today

- **Model wrappers** for U-Shape Transformer and CE-VAE (`divevision/models/`,
  `divevision/src/models/`), sharing a common `AbstractModel` interface
  (`divevision/src/models/abstract_model.py`).
- **A benchmark pipeline** (`divevision/src/test.py`) that runs each model against the LSUI and
  UIEB datasets, computes SSIM/PSNR metrics, and logs runs to MLflow.
- **A FastAPI server** (`divevision/src/app/main.py`) backed by Supabase (auth, photo storage,
  a `photos` table — see `AGENTS.md`). Endpoints: `/signup/` and `/login/`; an authenticated
  `POST /image/` that runs the U-Shape Transformer on an uploaded image, returns the enhanced
  PNG, and persists the original/processed photos plus a `photos` row; `DELETE /photos/{id}/`
  and `DELETE /account/` (full GDPR account erasure); and a shared-secret-gated
  `POST /leaderboard/` used by the benchmark pipeline to record scores. This is the seed of the
  "serve a model to a client" mobile-app goal above — it is not yet wired up to any mobile
  client.
- **Tests** for the models and the FastAPI app (`divevision/test/`).

## Roadmap (not implemented yet)

- Training a model from scratch (the README previously implied this existed — it does not; only
  inference over pretrained checkpoints is implemented).
- A dedicated web app.
- A real mobile app client consuming the FastAPI endpoint (or its successor).
- Social-network features and photo geolocation — explicitly out of scope until the above lands.

## Installation

### Prerequisites

- Python 3.12
- [Poetry](https://python-poetry.org/)
- Clone the repository

### Steps

1. `poetry install`
2. Download pretrained model weights: `./download_resources.sh`
   - This fetches only **model weights**, not the datasets (see below).
   - The CE-VAE checkpoint (`lsui-cevae-epoch119.ckpt`) is fetched from a GitHub Release asset
     on this repo (`cevae-checkpoint-v1` tag), and the U-Shape Transformer weights from Google
     Drive.
3. Download the datasets yourself — **this is not automated by any script in this repo**:
   - [LSUI dataset](https://bianlab.github.io/data.html) — expected at `divevision/data/LSUI/`,
     with `GT/` and `input/` subdirectories (see `divevision/src/datasets/lsui_dataset.py`).
   - [UIEB dataset](https://li-chongyi.github.io/proj_benchmark.html) — expected at
     `divevision/data/UIEB/`, with `raw-890/` and `reference-890/` subdirectories (see
     `divevision/src/datasets/uieb_dataset.py`). Academic use only, per the dataset's terms.
4. Try the notebook `divevision/notebooks/test_model.ipynb` to check that a model runs
   end-to-end, using the poetry environment.

## Running the benchmark

`divevision/src/test.py` runs both models against both datasets and logs metrics to MLflow.

1. Copy `.env_example` to `.env` and fill in the MLflow/Supabase/S3 variables it expects.
2. Start an MLflow tracking server: `./mlflow_server.sh` (reads `.env`, backs onto a Supabase
   Postgres DB and S3-compatible storage for run/artifact storage).
3. Run the benchmark: `poetry run python -m divevision.src.test`

## Running the FastAPI server

Fill in this app's own Supabase project variables (`SUPABASE_URL`, `SUPABASE_KEY`,
`SUPABASE_SERVICE_ROLE_KEY`, `LEADERBOARD_SHARED_SECRET`) in `.env` — see `AGENTS.md` for why
this is a separate project from the MLflow tracking backend's. Validate `supabase/migrations/`
locally with `supabase start` (requires Docker) before relying on them.

```
poetry run fastapi dev divevision/src/app/main.py
```

This exposes a form at `/` to upload an image, plus `/signup/`, `/login/`, an authenticated
`POST /image/` that returns the U-Shape Transformer's enhanced PNG output (and persists it —
see `AGENTS.md`), `DELETE /photos/{id}/`, `DELETE /account/`, and `POST /leaderboard/`. There is
no mobile client in this repository yet.

## Running tests

```
poetry run pytest
```

## Resources

- **U-Shape Transformer for Underwater Image Enhancement.** Peng L., Zhu C., Bian L., 2021.
  [Github](https://github.com/LintaoPeng/U-shape_Transformer_for_Underwater_Image_Enhancement) —
  [Paper](https://arxiv.org/abs/2111.11843)
- **CE-VAE: Capsule Enhanced Variational AutoEncoder for Underwater Image Enhancement.** Pucci R.,
  Martinal N., 2024.
  [Github](https://github.com/iN1k1/ce-vae-underwater-image-enhancement) —
  [Paper](https://arxiv.org/pdf/2406.01294v2)
