# Mistral Voice MCP Server

## Aim

MCP server for audio transcription using Mistral's **Voxtral Mini Transcribe 2** (`voxtral-mini-2602`).
Provides tools for transcribing audio files (single or batch), speaker diarization,
context biasing for domain-specific terms, and prompt templates for post-processing transcripts.

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
├── server.py          # FastMCP server, 12 tools + 1 prompt + 1 resource
├── transcriber.py     # Async Mistral API wrapper
├── workdir.py         # Work directory manager (input/output mirroring, context bias)
└── prompts.py         # Prompt templates for post-processing
```

## Key Dependencies

- `mistralai` - Official Mistral Python SDK
- `fastmcp` - FastMCP framework for MCP server

## Tools

| Tool | Purpose |
|------|---------|
| `mistral_set_workdir` | Set work directory (creates input/ and output/) |
| `mistral_get_workdir` | Show current workdir and status |
| `mistral_set_context_bias` | Set terms for transcription accuracy (text or file) |
| `mistral_get_context_bias` | Show configured bias terms |
| `mistral_clear_context_bias` | Clear all bias terms |
| `mistral_set_language` | Set transcription language (auto-disables timestamps for non-English) |
| `mistral_get_language` | Show current language and timestamp status |
| `mistral_list_inputs` | List audio files with transcription status |
| `mistral_transcribe_file` | Transcribe a single file |
| `mistral_transcribe_batch` | Transcribe all pending files |
| `mistral_list_transcriptions` | List completed transcriptions |
| `mistral_read_transcription` | Read a transcription result |
| `mistral_save_processed` | Save a cleaned/processed transcript as *_clean.md |

## Prompts

| Prompt | Purpose |
|--------|---------|
| `clean_transcript` | Clean raw transcript: remove filler words, fix grammar and logic flow, save with mistral_save_processed |

## Documentation

Reference docs in `docs/`:
- `01_offline_transcription.md` - API usage and code examples
- `02_voxtral_transcribe_2_announcement.md` - Model capabilities, benchmarks, pricing
- `03_voxtral_mini_transcribe_model.md` - Model reference card
