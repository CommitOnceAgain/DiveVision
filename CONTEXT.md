# DiveVision — Underwater Image Enhancement

DiveVision's current, code-backed scope is a research workflow for testing and comparing
underwater image enhancement models: paired benchmark datasets, model wrappers around
third-party architectures, quality metrics, and MLflow-based experiment tracking. The
mobile app / web app / FastAPI-serving strand mentioned in the README is out of scope for
this model — today it is a single unwrapped endpoint with no domain vocabulary of its own
yet (see "Out of scope" below).

## Language

### Images

**Degraded Image**:
An underwater photograph exhibiting color cast, haze, or contrast loss, before enhancement. The
input half of a paired sample in a Benchmark Dataset (e.g. LSUI's `input/`, UIEB's `raw-890/`).
_Avoid_: input image, raw image

**Reference Image**:
The clean, color-corrected counterpart to a Degraded Image, used as ground truth when scoring an
Enhanced Image (e.g. LSUI's `GT/`, UIEB's `reference-890/`).
_Avoid_: GT, ground truth, label, target

**Enhanced Image**:
The image an Enhancement Model produces by running its predict step on a Degraded Image.
_Avoid_: output image, restored image, prediction

### Models

**Enhancement Model**:
A wrapped, invokable model that maps a Degraded Image to an Enhanced Image, registered under a
short name (e.g. `"U-Shape"`, `"CVAE"`) and implementing the common `AbstractModel` interface
(`preprocessing` → forward pass → `postprocessing`). Currently two exist: U-Shape Transformer and
CE-VAE.
_Avoid_: model (ambiguous with Model Implementation below), network

**Model Implementation**:
The vendored third-party architecture code an Enhancement Model wraps unmodified
(`divevision/models/`), kept close to its upstream source so it stays diffable against the
original research repo. An Enhancement Model adapts one Model Implementation to the project's
common interface; the two are never the same object.
_Avoid_: model, wrapper (that's the Enhancement Model's role, not the implementation's)

**Checkpoint**:
The trained weights file an Enhancement Model loads into its Model Implementation before it can
run. Fetched separately from code, either via `download_resources.sh` (U-Shape Transformer) or a
config-referenced path (CE-VAE); an Enhancement Model without its Checkpoint present still
constructs but warns and runs with untrained weights.
_Avoid_: weights (fine as prose, but prefer Checkpoint as the noun for "the file")

### Data and evaluation

**Benchmark Dataset**:
A paired collection of Degraded Images and their corresponding Reference Images, used to evaluate
Enhancement Models. Currently LSUI and UIEB.
_Avoid_: dataset (only when precision matters — otherwise fine as shorthand)

**Evaluation Metric**:
A scoring function that compares an Enhanced Image against its Reference Image and produces a
numeric quality score. Currently SSIM and PSNR.
_Avoid_: score, measure

### Experiment tracking

**Experiment**:
The top-level MLflow container that groups every Benchmark Run for a given purpose (currently one
Experiment, `"Model testing"`, holds all of them).
_Avoid_: run (a Run is one execution inside an Experiment, not the container)

**Benchmark Run**:
One MLflow Run: the execution of one Enhancement Model against one full Benchmark Dataset,
producing per-batch and aggregate Evaluation Metric values plus elapsed-time figures logged to
MLflow.
_Avoid_: experiment (too broad — see Experiment above), test

## Out of scope

The FastAPI endpoint (`divevision/src/app/main.py`) and the mobile/web app it is meant to
eventually serve are early-stage and not modeled here. Today the endpoint is a single hard-coded
call to the U-Shape Enhancement Model with no client, no user or account concept, and no
persistence — there isn't yet a domain to name beyond the terms above. Revisit this file when
that strand grows real vocabulary (uploads, accounts, storage, etc.).
