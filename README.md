# Codex RAG Runner

Standalone RAG runner for NISM certification exam preparation. Indexes study
material files, retrieves relevant chunks via semantic search, and generates
grounded answers using an LLM backend (Claude Code CLI).

## Features

- **Source-grounded answers** — all responses cite retrieved chunks; unsupported
  questions are refused rather than guessed
- **Five study modes** — ask, teach, quiz, flashcards, revise
- **Interactive CLI** — local terminal REPL for direct Q&A
- **Web chat UI** — Gradio interface for desktop or mobile browsers
- **Google Colab notebook** — free cloud hosting with a public share link
- **GitHub Issues mode** — async processing via labeled GitHub issues
- **OCR cleaning** — strips picture placeholders and garbled text from converted
  workbook files before indexing

## Runtime Layout

- Source files: `./files`
- Vector index: `./index`
- Model cache: `./model-cache`
- Run logs: `./runs`

## Setup

```powershell
cd C:\Projects\codex_rag_runner
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
cp config.example.json config.json
```

Edit `config.json` if needed. The defaults work for the standard layout with
study files in the `files/` directory.

### LLM Backend

The runner uses Claude Code CLI as the LLM backend. Install it globally:

```powershell
npm install -g @anthropic-ai/claude-code
```

If `claude` is not on PATH, the runner falls back to `npx @anthropic-ai/claude-code`.

The backend can be configured in `config.json`:

```json
{
  "llm": {
    "backend": "auto",
    "timeout_seconds": 300
  }
}
```

Supported values for `backend`: `"auto"`, `"claude"`, `"claude-npx"`.

Supported values for `backend`: `"auto"`, `"claude"`, `"claude-npx"`, `"gemini"`.

For **Google Colab** or other cloud notebooks, set `"backend": "gemini"` and provide
a free API key from [Google AI Studio](https://aistudio.google.com/apikey) via the
`GEMINI_API_KEY` environment variable (or `llm.api_key` in config).

## Commands

### Web Chat UI (browser / mobile)

```powershell
python -m rag_runner --config config.json --web
```

Open `http://127.0.0.1:7860` in your browser. To access from a phone on the same
Wi‑Fi network:

```powershell
python -m rag_runner --config config.json --web --host 0.0.0.0
```

Then visit `http://<your-pc-ip>:7860` on your phone.

Use `--share` to get a temporary public Gradio link (same mechanism as Colab):

```powershell
python -m rag_runner --config config.json --web --share
```

### Google Colab (free, works on mobile)

1. Upload or open `notebooks/colab_chat.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Add your `GEMINI_API_KEY` under Colab **Secrets** (or paste it in the notebook).
3. Run all cells. Open the `*.gradio.live` link on your phone.

The notebook mounts **Google Drive** at `MyDrive/rag_study/` and saves:
- `files/` — study material
- `index/` — Chroma vector index (reused on later runs; no re-indexing)
- `model-cache/` — embedding model download cache

The first Colab run builds the index (a few minutes). Later runs load the saved index in seconds. After uploading new files, click **Rebuild index** in the chat UI.

The notebook uses the Gemini API because Claude Code CLI is not available in Colab.

### Interactive Study Mode (terminal)

```powershell
python -m rag_runner --config config.json --interactive
```

This starts a local REPL session. Type questions directly, switch study modes
with `/mode`, and see stats with `/stats`. Type `/help` for all commands.

### Build or Rebuild the Index

```powershell
python rag_index.py --config config.json --rebuild
```

The interactive mode builds the index automatically on startup, but you can
rebuild it manually with this command.

### GitHub Issues Mode

```powershell
# Process one batch of queued issues (dry run)
python -m rag_runner --config config.json --once --dry-run

# Poll continuously
python -m rag_runner --config config.json --loop
```

Set `github.token` or `github.token_env` in `config.json` with a valid GitHub
token.

## Study Modes

The runner reads the GitHub issue `Operation:` field, an optional `Study Mode:`
field, or the `/mode` REPL command, and changes the answer shape while keeping
the same source-grounding rules.

| Mode | Description |
|------|-------------|
| `ask` | Direct answer with explanation and certification-relevant terms |
| `teach` | Step-by-step tutor response with exam importance and self-checks |
| `quiz` | Practice questions plus cited answer key |
| `flashcards` | Concise Q/A cards with citations |
| `revise` | Revision sheet with must-know points, distinctions, and checklist |

## Answer Contract

Answers must be grounded in retrieved file chunks. When the configured folder
does not support an answer, the runner instructs the LLM to say so instead of
guessing.

## Running Tests

```powershell
python -m pytest tests/ -v
```
