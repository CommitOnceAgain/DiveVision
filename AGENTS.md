# AGENTS.md - Development Guidelines for DiveVision

This file contains essential information for agentic coding agents working in the DiveVision repository.

## Project Overview

DiveVision is an underwater image restoration system using deep learning models. The project combines PyTorch-based models with a FastAPI web service for processing underwater images.

**Tech Stack**: Python 3.12, Poetry, PyTorch, Lightning, FastAPI, OpenCV, MLFlow

## Development Commands

### Environment Setup
```bash
poetry install                    # Install all dependencies
./download_resources.sh          # Download model weights and resources
```

### Code Quality & Testing
```bash
# Code formatting and linting
poetry run black .               # Format code (88 char line limit)
poetry run isort .               # Sort imports
poetry run mypy .                # Type checking
poetry run bandit -r .           # Security linting

# Testing
poetry run pytest divevision/test/ --cov=divevision  # Run all tests with coverage
poetry run pytest divevision/test/test_models.py    # Run specific test file
poetry run pytest -k "test_model_pipeline"          # Run specific test by name
poetry run pytest divevision/test/test_models.py::TestModels::test_model_pipeline  # Run specific test method
```

### Docker & Deployment
```bash
docker build -t divevision .     # Build Docker image
./test-docker.sh                 # Test Docker container
./mlflow_server.sh               # Start MLFlow server
```

## Code Style Guidelines

### Formatting & Imports
- **Formatter**: Black with 88-character line limit
- **Import Sorting**: isort (standard imports first, then third-party, then local)
- **Quote Style**: Double quotes preferred
- **Type Hints**: Required for all function signatures and class attributes

### Naming Conventions
- **Classes**: PascalCase (e.g., `UShapeModelWrapper`, `AbstractModel`)
- **Functions/Variables**: snake_case (e.g., `preprocessing`, `model_ckpt`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MLFLOW_HOST`)
- **Private Members**: Leading underscore (e.g., `_model_registry`)

### Documentation
- Use triple-quoted docstrings for all public classes and methods
- Include type hints in function signatures
- Document parameters and return values in docstrings

## Architecture Patterns

### Abstract Base Classes
- All models inherit from `AbstractModel` in `src/models/abstract_model.py`
- All metrics inherit from `AbstractMetric` in `src/metrics/abstract_metric.py`
- All datasets inherit from `AbstractDataset` in `src/datasets/abstract_dataset.py`

### Registry Pattern
- Models are auto-registered via `__init_subclass__` mechanism
- Access models through the registry: `ModelRegistry.get_model("UShapeModel")`

### Wrapper Pattern
- External models (U-Shape Transformer, CE-VAE) are wrapped in consistent interfaces
- Wrappers handle preprocessing, model inference, and postprocessing

### Pipeline Pattern
- Standard pipeline: preprocessing → forward → postprocessing
- Each step is clearly separated and testable

## File Organization

```
divevision/
├── src/
│   ├── app/main.py              # FastAPI application entry point
│   ├── models/                  # Model wrappers and interfaces
│   ├── metrics/                 # Evaluation metrics
│   └── datasets/                # Dataset handlers
├── models/                      # External model implementations
│   ├── UShapeTransformer/       # U-Shape model code
│   └── CEVAE/                   # CE-VAE model code
└── test/                        # Test suite
```

## Testing Guidelines

### Test Structure
- Tests located in `divevision/test/`
- Use class-based test organization (e.g., `TestModels`)
- Parametrized tests with `@pytest.mark.parametrize` for multiple scenarios

### Test Writing
- Write unit tests for all model wrappers and metrics
- Test preprocessing and postprocessing pipelines
- Include integration tests for the FastAPI endpoints
- Use context managers for test setup and teardown

### Running Tests
- Always run tests before committing: `poetry run pytest divevision/test/`
- Use coverage reporting: `--cov=divevision --cov-report=html`
- Run specific tests during development to speed up iteration

## Model Development

### Adding New Models
1. Create wrapper class inheriting from `AbstractModel`
2. Implement required methods: `load_model`, `preprocess`, `forward`, `postprocess`
3. Add model to registry via auto-registration
4. Write comprehensive tests in `test_models.py`
5. Update documentation and API endpoints

### Model Configuration
- Use dataclasses for model configuration
- Support both local and remote model loading
- Handle model versioning and checkpointing

## API Development

### FastAPI Guidelines
- Use Pydantic models for request/response validation
- Include proper error handling and status codes
- Add OpenAPI documentation for all endpoints
- Support async operations where appropriate

### Endpoint Structure
- Health check endpoint: `/health`
- Model info endpoint: `/models`
- Image processing endpoint: `/process`
- Batch processing endpoint: `/batch-process`

## Dependencies & Environment

### Key Dependencies
- PyTorch 2.4.1+ for deep learning
- Lightning 2.5.0+ for training utilities
- FastAPI for web service
- OpenCV for image processing
- MLFlow for experiment tracking

### Environment Configuration
- Use `.env` file for local configuration
- See `.env_example` for required environment variables
- Support for Supabase integration
- MLFlow server configuration

## Security & Quality

### Security
- Run `poetry run bandit -r .` before committing
- Never commit secrets or API keys
- Validate all user inputs in API endpoints

### Code Quality
- All code must pass `black`, `isort`, and `mypy` checks
- Maintain test coverage above 80%
- Use type hints consistently
- Follow PEP 8 guidelines (enforced by Black)

## Git Workflow

### Committing Changes
1. Run all quality checks: `black .`, `isort .`, `mypy .`
2. Run tests: `poetry run pytest divevision/test/`
3. Stage changes and commit with descriptive message
4. Push and create PR if working on feature branch

### Branch Naming
- Use descriptive branch names: `feature/add-new-model`, `fix/api-endpoint`
- Keep branches focused on single features or fixes

## Docker & Deployment

### Docker Guidelines
- Use multi-stage builds for production
- Include all system dependencies for OpenCV
- Expose port 8000 for FastAPI
- Use non-root user for security

### Testing Docker
- Run `./test-docker.sh` to validate container
- Test API endpoints in container environment
- Verify model loading and inference

## MLFlow Integration

### Experiment Tracking
- Log model metrics and parameters
- Save model artifacts and checkpoints
- Use consistent naming conventions for experiments

### Server Setup
- Run `./mlflow_server.sh` for local development
- Configure remote tracking server for production
- Set proper URI in environment variables