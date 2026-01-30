# Code Gen Offline — Open WebUI + LangChain Agent

This project provides a **fully local, production-style setup** for running your very own ChatGPT Wrapper behind **Open WebUI**, giving you a ChatGPT-like interface backed by your own LLM backed by ChatGPT.

Isn't this ChatGPT with extra steps? Of Course Not!

It's possible to host your own tooling and open source LLM model from huggingface, and allow it to use whatever tools created in your server:
```
Browser
   ↓
Open WebUI (Frontend)
   ↓
LangChain Agent API (FastAPI)
   ↓
LLMs + Tools + RAG + Workflows
```

---

# 🗂 Project Structure

```
code-gen-offline/
│
├── docker-compose.yml   # Service orchestration
├── Dockerfile           # Agent server image
├── requirements.txt     # Python dependencies
├── app.py               # FastAPI + LangChain agent
├── .env                 # API keys & secrets (not committed)
└── README.md
```

---

# ⚙️ Prerequisites

* Docker Desktop (macOS / Linux / Windows)
* Docker Compose v2+

`requirements.txt` contains all the Python-sepcific tooling. This project is developed using Python 3.13 and virtual environements!

```
pip install -r requirements.txt
```

---

# 🔐 Environment Setup

The curreently supported LLM is OpenAI via the fastAPI server. This server needs an OPEN_AI_API key in order to serve the OpenWebui interface.

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
```

# 🏗 Build & Run

From the project root:

```bash
docker compose up --build -d
```

Check service status:

```bash
docker compose ps
```

You should see:

* `agent` → running on port **8000**
* `open-webui` → running on port **3001**


## Open WebUI Setup

1. Open your browser:

```
http://localhost:3001
```

Inside Open WebUI:

```
Avatar → Admin Panel → Settings → Connections → OpenAI → Manage → + Add
```

2. Fill in:

```
Name: Local LangChain Agent
Base URL: http://agent:8000/v1
API Key: YOUR_API_KEY_HERE_PLZ
```

3. Save!

If your model does not automatically appear:

```
Workspace → Models → + Add Model
```

It's called langchain-agent. It should appear in the models menu.

```
Model ID: langchain-agent
Provider: Local LangChain Agent
```

Save.


## The Server
The FastAPI server can be found at

```
http://localhost:8000
```
