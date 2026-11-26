# Distributed Memory Grid for Real-Time LLM Inference

Distributed Memory Grid (DMG) is a project to explore running large language model inference across multiple machines. Its intended approach is to split model layers between worker nodes, pass intermediate activations between stages, and expose inference and cluster controls through an API and web interface.

The repository contains frontend and backend foundations for that design. The current master and worker processes do not yet implement a functioning distributed inference cluster; see [Current status](#current-status) before trying the orchestration components.

## Objectives and architecture

The architecture specification in [`docs/architecture/overview.md`](docs/architecture/overview.md) describes the project goals:

- Use pipeline/model-layer parallelism to distribute a model across machines whose individual memory may be insufficient for the full model.
- Separate coordination (worker registration, heartbeats, and partition assignment) from data transfer (intermediate model activations).
- Provide an API gateway for inference requests and streamed output, with a web frontend for cluster and job views.

The repository currently has these corresponding foundations:

```text
Browser (React + Vite)             [routes; no API calls yet]

FastAPI gateway (backend/api) ----> SQLite user/job persistence

Master/controller + scheduler      [standalone scaffold]
Worker process + executor          [standalone scaffold]
proto/cluster.proto                 [declared contracts; not wired into these processes]
```

The API application mounts its routes under `/api` and exposes a public `/health` endpoint. The master, worker, scheduler, executor, generation, and tensor communication modules are separate foundations, not a fully connected runtime. In particular, the current controller and worker entry points do not start gRPC servers or communicate with each other.

## Technology stack

- **Backend:** Python, FastAPI, Pydantic Settings, SQLAlchemy, Alembic, PyTorch, Transformers, gRPC/protobuf.
- **Frontend:** React 18, Vite 8, React Router 7, Tailwind CSS 3, Lucide React.
- **Persistence:** SQLite by default, with SQLAlchemy and Alembic migrations.
- **Tests:** pytest for backend unit tests; the frontend package currently provides a production build command but no test script.
- **Containers:** Docker Compose definitions for a master process and two worker processes.

## Repository structure

```text
.
├── backend/
│   ├── api/             # FastAPI app, authentication, and REST/WebSocket routes
│   ├── core/            # Settings and database setup
│   ├── distributed/     # Tensor communication and pipeline-parallel foundations
│   ├── inference/       # Model loading, generation, and KV-cache modules
│   ├── master/          # Worker registry, controller, and scheduler
│   ├── models/          # SQLAlchemy models
│   ├── monitoring/      # System telemetry and metrics
│   ├── schemas/         # API schemas
│   ├── services/        # Authentication, user, and inference-job services
│   └── worker/          # Worker process and partition executor
├── docker/
│   └── Dockerfile.backend
├── docs/
│   ├── architecture/overview.md
│   ├── development-log.md
│   └── frontend-architecture.md
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── services/
│   └── package.json
├── migrations/          # Alembic environment and database revisions
├── proto/cluster.proto  # Declared gRPC service/message contracts
├── tests/unit/          # Backend unit tests
├── .env.example
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Backend setup

The Dockerfile targets Python 3.11. From the repository root, create and activate a virtual environment, then install the declared Python dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

On macOS or Linux, activate with `source .venv/bin/activate` and copy the environment file with `cp .env.example .env`.

### Environment configuration

The root [`.env.example`](.env.example) contains backend settings, including the API host/port, CORS origins, database URL, and JWT options. The example JWT secret is deliberately a placeholder and is rejected by the settings validator. Replace it in `.env` with a locally generated secret of at least 32 characters. For example:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

The checked-in example uses `sqlite:///./data/inference_jobs.db`. `DATABASE_URL` can be changed to another SQLAlchemy database URL. `CORS_ORIGINS` must contain one or more explicit HTTP(S) origins; the example allows the Vite development origin `http://localhost:5173`.

The frontend's optional `VITE_API_BASE_URL` is separate from the backend `.env.example`. The frontend API service defaults it to `http://localhost:8000/api`; frontend pages do not currently call that service.

### Database and API gateway

With a valid backend `.env` in place, run database migrations from the repository root:

```powershell
python -m alembic upgrade head
```

Then start the FastAPI gateway:

```powershell
python -m backend.api.main
```

The default settings bind the API to `0.0.0.0:8000`; the local health check is `http://localhost:8000/health`. Settings are loaded from `.env` and validated at startup, including the JWT secret.

### Master, workers, and Docker Compose

The master and worker entry points can be run separately:

```powershell
python -m backend.master.controller
python -m backend.worker.worker_node --worker-id worker-1 --port 50052 --master-address localhost:50051
```

These are currently process-loop scaffolds: the controller runs a periodic registry check, and the worker logs local CPU/memory heartbeats. They do not bind the declared gRPC ports or exchange RPCs. The gRPC services and messages in [`proto/cluster.proto`](proto/cluster.proto) are contracts, not evidence that these RPCs are implemented.

The current Compose file can be started from the repository root with:

```powershell
docker compose up --build
```

It launches the master controller and two worker processes. Although the Compose file publishes port `8000`, its `master` command overrides the Dockerfile's API command, so Compose does **not** currently start the FastAPI gateway. The Compose setup is therefore not an end-to-end inference deployment.

## Frontend setup

Vite 8 requires Node.js `^20.19.0` or `>=22.12.0`. From the frontend directory:

```powershell
cd frontend
npm install
npm run dev
```

The development server prints its local URL (normally `http://localhost:5173`). To create a production build:

```powershell
npm run build
```

The frontend routes are `/` (Overview), `/workers`, `/models`, `/inference`, and `/jobs`. Unknown paths redirect to `/`. The dark theme is the default, with a persistent light/dark toggle.

## Backend API currently exposed

The FastAPI application currently declares these routes:

- `GET /health` — public gateway health response.
- `POST /api/auth/register` and `POST /api/auth/login` — account registration and bearer-token login.
- `POST /api/inference/generate` and `GET /api/inference/jobs/{job_id}` — create and retrieve an authenticated user's job record.
- `GET /api/workers`, `GET /api/cluster/status`, and `GET /api/metrics` — authenticated monitoring routes. Worker details and several reported values are sample data, not connected cluster measurements.
- `POST /api/models/load` — currently returns `501 Not Implemented`.
- `WS /api/ws/inference/{job_id}` — currently emits a fixed demonstration token sequence. It is not connected to the generation engine, does not authenticate the socket, and does not verify job ownership.

These routes are backend scaffolding and are not integrated into the frontend. The frontend displays a neutral “Not connected” status and does not claim live worker, model, job, or inference data.

## Tests

Backend unit tests are in `tests/unit/` and use pytest. After installing the Python requirements and configuring a valid `.env`:

```powershell
python -m pytest tests/unit
```

The frontend has no test command in `frontend/package.json`; use `npm run build` to validate its production bundle.

## Current status and next integration phases

**Present:** a routed React frontend with responsive navigation and theme persistence; a FastAPI application with authentication routes, SQLite-backed user/job models, migrations, and health check; backend modules for scheduling, partition execution, generation, and telemetry; and protobuf service definitions.

**Still placeholder or disconnected:** frontend API/auth integration; live worker/model/job views; actual model dispatch through the master and workers; active gRPC service implementations; end-to-end generated-token streaming; and Compose startup of the API gateway.

The following are proposed next phases based on those repository gaps, not a committed schedule:

1. Implement and connect the master/worker gRPC control and execution services defined in `proto/cluster.proto`.
2. Connect API job lifecycle and the inference generation pipeline to real model execution and authenticated streaming.
3. Replace sample worker, cluster, and metrics responses with live runtime state; add WebSocket authentication and job-ownership checks.
4. Integrate frontend authentication and API calls, then populate worker, model, inference, and job-history pages from backend data.
5. Update Compose to launch and configure the API gateway alongside the coordinator and workers, with suitable health checks.
