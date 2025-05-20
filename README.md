# Mitrailleuse

> **Multi‑provider AI request launcher & batch‑runner** supporting **OpenAI**, **DeepSeek**, and **DeepL**, with a gRPC façade, Dockerised runtime, and sampling‑aware task folders.

> Mitrailleuse is an extensible micro‑service that orchestrates high‑throughput requests to multiple generative‑AI providers (OpenAI, DeepSeek, DeepL) while providing a consistent gRPC interface, pluggable adapters, dynamic prompt mapping, and first‑class batch support.
---

## ✨ Key features

| Capability            | Details                                                                                                                     |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 🗂️ Task folders      | Every call lives under `tasks/<user_id>/<api>_<task>_<timestamp>/` (inputs, outputs, logs, config).                         |
| 📡 Multi‑provider     | Switch between `openai`, `deepseek`, `deepl` by setting `api` in the task JSON or `--api` on the CLI.                       |
| 🚀 gRPC service       | Exposes `CreateTask`, `SendSingle`, `CreateBatch`, `CheckBatchStatus`, `DownloadBatchResults`. Reflection & health enabled. |
| 🐳 Docker‑ready       | Single `mitrailleuse-grpc` image builds from `Dockerfile`, spun up via `docker‑compose`.                                    |
| 🔍 Sampling           | `config.general.sampling` lets you process the first *N* lines for quick dry‑runs.                                          |
| 🧩 Adapters           | New providers drop in via `Adapters` registry with `send_single` / `send_batch`.                                            |
| 📜 CLI Interface      | New command-line interface for task management and execution.                                                               |
| 🔄 File Flattening    | Automatic flattening of nested JSON structures for consistent processing.                                                   |
| 🔍 Similarity Check   | Configurable similarity checking to prevent duplicate responses.                                                            |
| 💾 Caching            | In-memory and file-based caching for improved performance.                                                                  |

---

## 🏗️ Architecture
```
           +------------------------------+
           |           Clients            |
           |  (Postman, grpcurl, CLI)     |
           +------------------------------+
                        │ gRPC
                        ▼
           +------------------------------+
           |  Mitrailleuse gRPC Server    |
           |  • Health + Reflection       |
           |  • RequestService            |
           +-------------┬----------------+
                         │ chooses adapter
           ┌─────────────┴───────────────┐
           │                             │
+------------------+         +------------------+
|  OpenAIAdapter   |         | DeepSeekAdapter  |
|  – batch + chat  |         |  – chat          |
+------------------+         +------------------+
           │                             │
           └─────────────┬───────────────┘
                         │
               +------------------+
               |  DeepLAdapter    |
               |  – translate     |
               +------------------+
```

## Quick start

### Prerequisites

* Python 3.12+
* Docker 24.x (optional but recommended)
* Provider API keys in config.json

### 1 · Clone & build

```bash
# clone
git clone https://github.com/theFellandes/mitrailleuse.git
cd mitrailleuse

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in development mode
pip install -e .

# Install dependencies
pip install -r requirements.txt

# Generate gRPC stubs
python -m grpc_tools.protoc -I. \
  --python_out=. --grpc_python_out=. mitrailleuse/proto/mitrailleuse.proto

# build the gRPC server image (if using Docker)
docker compose build mitrailleuse-grpc
```

### 2 · Set API keys

Create `.env` in the repo root:

```dotenv
OPENAI_API_KEY=sk‑...
DEEPSEEK_API_KEY=dsk‑...
DEEPL_API_KEY=dpa‑...
```

### 3 · Run the stack

#### Option 1: Run directly with Python
```bash
# Start the gRPC server
python -m mitrailleuse.infrastructure.grpc.server
```

#### Option 2: Run with Docker
```bash
docker compose up -d mitrailleuse-grpc
```

The server listens on **`localhost:50051`** (gRPC, plaintext).

### 4 · Smoke test

```bash
grpcurl -plaintext localhost:50051 list
```

Should list `mitrailleuse.MitrailleuseService`.

---

## 🔧 Configuration files

### `config.json` template

```jsonc
{
  "general": {
    "verbose": true,
    "sampling": { "enable_sampling": true, "sample_size": 100 },
    "check_similarity": true,
    "similarity_settings": {
      "similarity_threshold": 0.8,
      "cooldown_period": 300,
      "max_recent_responses": 100
    }
  },

  "openai": {
    "prompt": "input_text",
    "system_instruction": {
      "is_dynamic": true,
      "system_prompt": "instructions"
    },
    "api_information": { "model": "gpt-4o-mini" },
    "batch": {
      "is_batch_active": true,
      "batch_size": 20,
      "batch_check_time": 120,
      "combine_batches": false
    }
  },

  "deepseek": {
    "prompt": "input_text",
    "system_instruction": {
      "is_dynamic": true,
      "system_prompt": "instructions"
    },
    "api_information": { "model": "deepseek-chat" },
    "batch": {
      "is_batch_active": true,
      "batch_size": 20,
      "batch_check_time": 120,
      "combine_batches": false
    }
  },

  "deepl": {
    "api_key": "${DEEPL_API_KEY}",
    "target_lang": "lang",   // field name inside input JSON
    "text": "content"        // field with text to translate
  }
}
```

---

## 🖥️ CLI Usage

The new CLI provides a more intuitive interface for task management:

```bash
# Make sure you're in the project root directory
cd mitrailleuse

# Create a new task
python -m mitrailleuse.main create --user-id user1 --api-name openai --task-name demo

# List available tasks
python -m mitrailleuse.main list --user-id user1

# Get task information
python -m mitrailleuse.main get --task-path /path/to/task

# Execute a task
python -m mitrailleuse.main execute --user-id user1 --task-folder /path/to/task
```

### CLI Features

* Task creation with custom configuration
* Task listing and status monitoring
* Batch job tracking and status updates
* Support for all providers (OpenAI, DeepSeek, DeepL)
* Automatic file flattening and format conversion
* Similarity checking to prevent duplicate responses
* Caching for improved performance

---

## 🗄️ Directory structure (runtime)

```
mitrailleuse/
├─ mitrailleuse/
│   ├─ proto/
│   │   ├─ mitrailleuse.proto
│   │   ├─ mitrailleuse_pb2.py
│   │   └─ mitrailleuse_pb2_grpc.py
│   ├─ infrastructure/
│   │   ├─ grpc/
│   │   │   └─ server.py
│   │   └─ ...
│   └─ ...
├─ tasks/
│   └─ alice/
│       └─ openai_demo_03_05_2025_142530/
│           ├─ config/config.json
│           ├─ inputs/
│           │   ├─ backup/
│           │   └─ ...
│           ├─ outputs/
│           ├─ cache/
│           └─ logs/
│               ├─ app.log
│               └─ batch.log
└─ ...
```

---

## 🛰️ gRPC API reference (proto excerpt)

```proto
service MitrailleuseService {
  rpc CreateTask           (CreateTaskRequest)           returns (TaskEnvelope);
  rpc SendSingle           (SingleRequest)               returns (SingleResponse);
  rpc CreateBatch          (BatchRequest)                returns (BatchJob);
  rpc CheckBatchStatus     (BatchStatusRequest)          returns (BatchJob);
  rpc DownloadBatchResults (BatchResultRequest)          returns (stream BatchLine);
}
```

*Reflection* is enabled: Postman & `grpcurl` can auto‑discover.

---

## 🐳 Docker compose

```yaml
services:
  mitrailleuse-grpc:
    build: ./mitrailleuse/grpc
    ports:
      - "50051:50051"
    env_file: .env
    volumes:
      - ./tasks:/app/tasks
```

---

## 🧪 Development & testing

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
black . --check
```

Generate updated stubs:

```bash
python -m grpc_tools.protoc -I. \ 
  --python_out=. --grpc_python_out=. mitrailleuse.proto
```

---

## 📜 License

MIT – see `LICENSE`.

---

## 🗺️ Roadmap

* [ ] DeepSeek batch support when API ships
* [ ] Streaming gRPC endpoint
* [ ] Web dashboard for task monitoring
* [ ] Kubernetes Helm chart
* [ ] Enhanced similarity checking with more algorithms
* [ ] Support for more AI providers

Contributions welcome — see `CONTRIBUTING.md`.
