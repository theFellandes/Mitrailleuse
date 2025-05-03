# Mitrailleuse

> **Multi‑provider AI request launcher & batch‑runner** supporting **OpenAI**, **DeepSeek**, and **DeepL**, with a gRPC façade, Dockerised runtime, and sampling‑aware task folders.

---

## ✨ Key features

| Capability            | Details                                                                                                                     |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 🗂️ Task folders      | Every call lives under `tasks/<user_id>/<api>_<task>_<timestamp>/` (inputs, outputs, logs, config).                         |
| 📡 Multi‑provider     | Switch between `openai`, `deepseek`, `deepl` by setting `api` in the task JSON or `--api` on the CLI.                       |
| 🚀 gRPC service       | Exposes `CreateTask`, `SendSingle`, `CreateBatch`, `CheckBatchStatus`, `DownloadBatchResults`. Reflection & health enabled. |
| 🐳 Docker‑ready       | Single `mitrailleuse-grpc` image builds from `Dockerfile`, spun up via `docker‑compose`.                                    |
| 🔍 Sampling           | `config.general.sampling` lets you process the first *N* lines for quick dry‑runs.                                          |
| 🧩 Adapters           | New providers drop in via `Adapters` registry with `send_single` / `send_batch`.                                            |
| 📜 Standalone clients | `create_task.py`, `send_single.py`, `batch_workflow.py` for scripting and CI.                                               |

---

## 🏁 Quick start

### 1 · Clone & build

```bash
# clone
 git clone https://github.com/theFellandes/mitrailleuse.git
 cd mitrailleuse

# build the gRPC server image
 docker compose build mitrailleuse-grpc
```

### 2 · Set API keys

Create `.env` in the repo root:

```dotenv
OPENAI_API_KEY=sk‑...
DEEPSEEK_API_KEY=dsk‑...
DEEPL_API_KEY=dpa‑...
```

### 3 · Run the stack

```bash
docker compose up -d mitrailleuse-grpc
```

The server listens on **`localhost:50051`** (gRPC, plaintext).

### 4 · Smoke test

```bash
grpcurl -plaintext localhost:50051 list
```

Should list `mitrailleuse.MitrailleuseService`.

---

## 🔧 Configuration files

### `config.json` template

```jsonc
{
  "general": {
    "verbose": true,
    "sampling": { "enable_sampling": true, "sample_size": 100 }
  },

  "openai": {
    "prompt": "input_text",
    "system_instruction": {
      "is_dynamic": true,
      "system_prompt": "instructions"
    },
    "api_information": { "model": "gpt-4o-mini" }
  },

  "deepseek": {
    "prompt": "input_text",
    "system_instruction": {
      "is_dynamic": true,
      "system_prompt": "instructions"
    },
    "api_information": { "model": "deepseek-chat" }
  },

  "deepl": {
    "api_key": "${DEEPL_API_KEY}",
    "target_lang": "lang",   // field name inside input JSON
    "text": "content"        // field with text to translate
  }
}
```

---

## 🖥️ Standalone CLI usage

```bash
# 1. create a task folder
python create_task.py --user alice --api openai --name demo \
       --config configs/config.json

# 2. send a single request
python send_single.py --user alice --api deepseek \
       --messages inputs/hello.json

# 3. run and download an OpenAI batch
python batch_workflow.py --user alice --api openai \
       --jsonl inputs/batch_inputs.jsonl
```

> **Note:** DeepSeek mirrors OpenAI’s chat endpoint, but currently has *no* batch API; `batch_workflow` exits early for `--api deepseek`. DeepL supports single‑shot translation only.

---

## 🗄️ Directory structure (runtime)

```
mitrailleuse/
├─ tasks/
│   └─ alice/
│       └─ openai_demo_03_05_2025_142530/
│           ├─ config/config.json
│           ├─ inputs/...
│           ├─ outputs/...
│           └─ logs/app.log
└─ ...
```

---

## 🛰️ gRPC API reference (proto excerpt)

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

## 🐳 Docker compose

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

## 🧪 Development & testing

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

## 📜 License

MIT – see `LICENSE`.

---

## 🗺️ Roadmap

* \[ ] DeepSeek batch support when API ships
* \[ ] Streaming gRPC endpoint
* \[ ] Web dashboard for task monitoring
* \[ ] Kubernetes Helm chart

Contributions welcome — see `CONTRIBUTING.md`.
