# Mistral Voice MCP Server

## Aim

MCP server for audio transcription using Mistral's **Voxtral Mini Transcribe 2** (`voxtral-mini-2602`).
Provides a session-based workflow: drop audio in inbox, transcribe to staging, review,
then finalize into a named recording directory with merged transcript.

## Environment & Dependency Management

**Everything MUST be managed through [uv](https://docs.astral.sh/uv/).**

- **Always use `uv add <package>`** to add dependencies. Never `uv pip install` or `pip install`.
- Run with `uv run`.
- `pyproject.toml` is the single source of truth for dependencies.

## Running

```bash
# Start the MCP server (stdio transport)
uv run mistral-voice-mcp

# Or directly
uv run python src/mistral_voice_mcp/server.py
```

Requires `MISTRAL_API_KEY` environment variable.

## Testing

```bash
# Unit tests (no API calls)
uv run pytest

# Integration tests (calls real Mistral API, costs money)
uv run pytest -m integration
```

## Project Structure

```
src/mistral_voice_mcp/
├── __init__.py
├── server.py          # FastMCP server, 13 tools + 1 prompt + 1 resource
├── transcriber.py     # Async Mistral API wrapper
├── staging.py         # Staging sessions, merge, finalize, slugify
├── workdir.py         # Work directory manager (inbox, staging, recordings, context bias)
└── prompts.py         # Prompt templates for post-processing
```

## Key Dependencies

- `mistralai` - Official Mistral Python SDK
- `fastmcp` - FastMCP framework for MCP server

## Workflow

1. `mistral_set_workdir` → creates `inbox/` and `.staging/`
2. Drop audio files in `inbox/`
3. `mistral_create_session` → copies files to staging
4. `mistral_transcribe` → transcribes all files in session
5. `mistral_read_staging_transcript` → LLM reads transcript, picks a name
6. `mistral_finalize(session_id, "descriptive name")` → creates named directory, merges transcript, cleans up
7. (optional) `clean_transcript` prompt + `mistral_save_processed` → writes `transcript_clean.md`

### Output structure

```
workdir/
  descriptive-name/
    audio1.mpeg
    audio2.mpeg           (if multi-file)
    transcript.md         (merged, with # Part: headers for multi-file)
    transcript_clean.md   (optional, after cleaning)
```

## Tools

| Tool | Purpose |
|------|---------|
| `mistral_set_workdir` | Set work directory (creates inbox/ and .staging/) |
| `mistral_get_workdir` | Show current workdir and status |
| `mistral_set_context_bias` | Set terms for transcription accuracy (text or file) |
| `mistral_get_context_bias` | Show configured bias terms |
| `mistral_clear_context_bias` | Clear all bias terms |
| `mistral_set_language` | Set transcription language (auto-disables timestamps for non-English) |
| `mistral_get_language` | Show current language and timestamp status |
| `mistral_list_inbox` | List audio files in inbox/ |
| `mistral_create_session` | Stage inbox files for transcription |
| `mistral_list_sessions` | List active staging sessions |
| `mistral_transcribe` | Transcribe all files in a staging session |
| `mistral_read_staging_transcript` | Read merged transcript from staging (before naming) |
| `mistral_finalize` | Finalize session into named recording directory |
| `mistral_list_recordings` | List finalized recording directories |
| `mistral_read_transcript` | Read transcript from a finalized recording |
| `mistral_save_processed` | Save cleaned transcript as transcript_clean.md |

## Prompts

| Prompt | Purpose |
|--------|---------|
| `clean_transcript` | Clean raw transcript: remove filler words, fix grammar and logic flow, save with mistral_save_processed |

## Documentation

Reference docs in `docs/`:
- `01_offline_transcription.md` - API usage and code examples
- `02_voxtral_transcribe_2_announcement.md` - Model capabilities, benchmarks, pricing
- `03_voxtral_mini_transcribe_model.md` - Model reference card
