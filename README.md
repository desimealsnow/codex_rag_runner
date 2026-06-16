# Codex RAG Runner

Standalone GitHub-triggered RAG runner for study questions over files stored on
the VM. This project is intentionally separate from `codex_automation`; the
existing automation runner is not imported or modified.

## Runtime Layout

- Source files: `/scratch/rnerolu/codex-rag/files`
- Vector index: `/scratch/rnerolu/codex-rag/index`
- Model cache: `/scratch/rnerolu/codex-rag/model-cache`
- Venv: `/scratch/rnerolu/codex-rag/venv`

## Setup

```bash
cd /scratch/rnerolu/codeshare/projects/codex_rag_runner
python3.11 -m venv /scratch/rnerolu/codex-rag/venv
/scratch/rnerolu/codex-rag/venv/bin/pip install -r requirements.txt
cp config.example.json config.json
```

Edit `config.json` with the GitHub repo and token. You can either set
`github.token` directly in the ignored local config, or set `github.token_env`
to the name of an environment variable that contains the token.

## Commands

Run these from the RAG runner repo:

```bash
cd /scratch/rnerolu/codeshare/projects/codex_rag_runner
/scratch/rnerolu/codex-rag/venv/bin/python -m rag_index --config config.json --rebuild
/scratch/rnerolu/codex-rag/venv/bin/python -m rag_runner --config config.json --once --dry-run
/scratch/rnerolu/codex-rag/venv/bin/python -m rag_runner --config config.json --loop
```

Or run them from any directory with absolute paths:

```bash
/scratch/rnerolu/codex-rag/venv/bin/python /scratch/rnerolu/codeshare/projects/codex_rag_runner/rag_index.py --config /scratch/rnerolu/codeshare/projects/codex_rag_runner/config.json --rebuild
cd /scratch/rnerolu/codeshare/projects/codex_rag_runner && /scratch/rnerolu/codex-rag/venv/bin/python -m rag_runner --config /scratch/rnerolu/codeshare/projects/codex_rag_runner/config.json --once --dry-run
cd /scratch/rnerolu/codeshare/projects/codex_rag_runner && /scratch/rnerolu/codex-rag/venv/bin/python -m rag_runner --config /scratch/rnerolu/codeshare/projects/codex_rag_runner/config.json --loop
```

## Answer Contract

Answers must be grounded in retrieved file chunks. When the configured folder
does not support an answer, the runner instructs Codex to say so instead of
guessing.
