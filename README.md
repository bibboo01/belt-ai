# Belt AI

Belt AI is a production-oriented belt inspection system for detecting belt defects with YOLO11 and storing inspection results in PostgreSQL.

The current system supports three detection classes:

- `dogear-belt`
- `good-belt`
- `splice-belt`

The runtime API receives an image, validates the camera and production model, runs YOLO inference, saves the original/result images, writes detection and inspection data to PostgreSQL, and returns the inspection result.

## Current Production Model

| Item | Value |
|---|---|
| Model | YOLO11 |
| Version | ODBv54 |
| Model ID | 2 |
| Status | PRODUCTION |
| Production file | `models/production/best.pt` |
| Classes | `dogear-belt`, `good-belt`, `splice-belt` |

## System Flow

```text
Camera / Image Upload
        |
        v
    FastAPI /detect
        |
        v
 DetectionService
        |
        +--> Camera validation
        +--> Image validation / decode
        +--> Production model validation
        |
        v
   YOLO11 ODBv54
        |
        v
    Detections
        |
        +--> Save original image
        +--> Save result image
        +--> Write detection records
        +--> Write inspection_result
        +--> Update image status = DONE
        |
        v
     PostgreSQL
```

## Project Structure

```text
belt-ai/
│
├── app/                              # Runtime backend and inference
│   ├── main.py                       # FastAPI application / API endpoints
│   ├── config.py                     # Application configuration / .env settings
│   ├── errors.py                     # Belt AI custom errors and error codes
│   ├── schemas.py                    # API request/response schemas
│   ├── logging_config.py             # Application logging configuration
│   ├── model_manager.py              # Production model loading and validation
│   ├── inference.py                  # YOLO inference and result summarization
│   ├── detection_service.py          # End-to-end detection orchestration
│   └── database/
│       └── repository.py              # PostgreSQL access layer / transactions
│
├── database/                         # Database assets, SQL, migrations, seeds
│
├── datasets/                         # Dataset versions used by training/evaluation
│   └── versions/
│
├── docs/                             # System documentation
│
├── evaluation/                       # Model evaluation and production gate
│   └── results/                      # Evaluation outputs
│
├── images/                            # Runtime image storage
│   ├── original/                     # Uploaded/captured source images
│   └── result/                       # Images with detection overlays
│
├── mlops/                             # ML lifecycle tracking and metadata
│   └── metadata/
│
├── models/                            # Model lifecycle storage
│   ├── archive/                      # Retired/previous models
│   ├── production/                   # Active production model
│   │   ├── best.pt
│   │   └── model.json
│   └── staging/                      # Models under evaluation
│
├── requirements/                     # Dependency requirement files
│
├── tests/                             # Automated tests and audit scripts
│   ├── unit/
│   │   ├── test_inference.py
│   │   ├── test_model_manager.py
│   │   └── test_database_repository.py
│   ├── test_detect.py                # API/integration test
│   ├── test_api_fault_injection.py   # API fault-injection tests
│   ├── test_recovery_audit.py        # Recovery/rollback tests
│   └── test_database_errors.py       # Database error audit script
│
├── training/                          # Model training code/configuration
│
├── .env                               # Local runtime configuration (do not commit)
├── .env.example                       # Configuration template
├── .gitignore                          # Git ignore rules
├── belt_ai_backup_before_mlops.sql    # Database backup created before MLOps changes
└── mlflow.db                          # Local MLflow tracking database
```

## Main Responsibilities

### `app/main.py`

FastAPI entry point. It exposes the HTTP API and delegates business logic to the detection service.

### `app/model_manager.py`

Controls the production model lifecycle at runtime. It loads the model from `models/production/`, validates the production registry information, checks the detection task and expected class mapping, and exposes the loaded model to inference.

### `app/inference.py`

Runs YOLO inference on a decoded NumPy frame and converts model output into a standard `InferenceResult` containing bounding boxes, confidence values, class counts, and the overall `GOOD`/`NG` result.

Current inspection rule:

```text
splice-belt or dogear-belt -> NG
no defect detection        -> GOOD
```

### `app/detection_service.py`

Orchestrates the complete request pipeline from image validation through model execution, file storage, database writes, and final response creation.

### `app/database/repository.py`

Contains the database access layer. It provides normal operations and transaction-safe operations for image, detection, inspection result, camera, and model registry data.

## Database

The current PostgreSQL database is `belt_ai` and the main application schema is `belt`.

Core tables include:

```text
belt.machine
belt.camera
belt.image
belt.detection
belt.inspection_result
belt.model_version
```

Important runtime relationships:

```text
camera
  |
  +--> image
          |
          +--> detection --> model_version
          |
          +--> inspection_result
```

The detection record stores `model_id`, which preserves model lineage for each detection.

## API

### Health

```http
GET /health
```

### Detection

```http
POST /detect?camera_id=1
```

Example with PowerShell:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/detect?camera_id=1" -F "image=@images/original/153_recovery_audit.jpg"
```

Example successful response shape:

```json
{
  "status": "success",
  "image": {
    "image_id": 174,
    "filename": "153_recovery_audit.jpg",
    "width": 1920,
    "height": 1080,
    "original_path": "images/original/174_153_recovery_audit.jpg",
    "result_path": "images/result/174_153_recovery_audit_result.jpg"
  },
  "model": {
    "model_id": 2,
    "name": "YOLO11",
    "version": "ODBv54"
  },
  "result": {
    "overall": "NG",
    "good_count": 0,
    "splice_count": 1,
    "dogear_count": 0,
    "ng_count": 1,
    "total_detections": 1
  }
}
```

## Environment

Current development environment:

- Windows
- Python 3.11.x
- FastAPI
- Uvicorn
- Ultralytics YOLO
- OpenCV
- NumPy
- SQLAlchemy
- PostgreSQL 17
- pytest
- MLflow (local tracking database)

The local development machine currently runs PyTorch on CPU.

## Running the Application

Use the virtual environment Python directly:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

## Testing

### Full test suite

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

### Inference unit tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_inference.py -v
```

### Model manager unit tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_model_manager.py -v
```

### Repository unit tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_database_repository.py -v
```

### Database error audit

`tests/test_database_errors.py` is currently structured as a database audit script rather than a pytest test module. Run it directly:

```powershell
.\.venv\Scripts\python.exe .\tests\test_database_errors.py
```

## Model Lifecycle

```text
Training
   |
   v
models/staging/
   |
   v
Evaluation / Production Gate
   |
   v
models/production/
   |
   v
app/model_manager.py
   |
   v
/detect
```

Older production models can be moved to `models/archive/` when they are retired.

## MLOps

The `mlops/` area is used for experiment tracking and model metadata.

Conceptually:

```text
Training
   |
   v
MLflow Tracking
   |
   +--> parameters
   +--> metrics
   +--> artifacts
   +--> run metadata
   |
   v
Evaluation
   |
   v
Model Registry / Production
```

`models/` stores model files, while `mlops/` stores information used to track and understand how models were trained, evaluated, and promoted.

## Current Status

Implemented and verified:

- FastAPI detection endpoint
- Production model loading through `ModelManager`
- YOLO11 ODBv54 inference
- Image/result file handling
- PostgreSQL image/detection/inspection writes
- Model lineage through `model_id`
- Transaction-safe repository methods
- Unit tests for inference, model manager, and repository
- API integration tests

In progress:

- Fault-injection and recovery test cleanup
- Production hardening
- Complete test suite stabilization
- Camera capture service
- Multi-camera / DB-driven camera configuration
- Expanded MLOps tracking and metadata
- Production deployment and monitoring

## Development Principles

The project is organized to keep responsibilities separated:

```text
app/         -> Runtime application
training/    -> Model training
evaluation/  -> Model evaluation
models/      -> Model lifecycle
datasets/    -> Dataset lifecycle
database/    -> Database assets
mlops/       -> ML tracking and metadata
tests/       -> Quality and regression protection
docs/        -> Documentation
```

This separation is intended to make the system easier to test, operate, and extend toward a multi-camera production environment.
