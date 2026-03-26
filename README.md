# Voxtract

Extract structured knowledge from voice recordings.

Voxtract is an MCP server that transcribes audio files using Mistral's Voxtral model and lets your LLM agent handle the post-processing — cleaning, restructuring, summarizing — all within your existing setup. No extra AI agents, no extra costs beyond Mistral transcription at **$0.003/minute**.

## Why Voxtract?

Dedicated voice-to-text products charge $80–180 for hardware plus monthly subscriptions for AI features. Voxtract takes a different approach:

- **Mistral handles transcription** — $0.003/min (~$0.18/hour). Speaker diarization, 13 languages, context biasing for domain terms.
- **Your LLM handles intelligence** — The agent you already use (Claude, GPT, etc.) does the post-processing. No extra API costs. Cleaning, restructuring, meeting notes, idea extraction — whatever you need.
- **MCP glues them together** — Voxtract runs as an [MCP server](https://modelcontextprotocol.io/) inside any MCP-compatible client (Claude Code, Claude Desktop, Cursor, Windsurf, or any agent framework supporting MCP). The LLM calls the tools directly, reads the transcript, and processes it in the same conversation.

The result: you record on your phone, drop the file in any project directory, and get organized markdown knowledge — for essentially free on top of what you already pay.

## How it works

```
You speak → Audio file → Voxtract transcribes → LLM cleans & structures → Organized knowledge
```

Voxtract uses a session-based workflow:

1. Drop audio files in a working directory
2. Voxtract transcribes them via Mistral and stages the results
3. The LLM reads the transcript, picks a descriptive name, and finalizes into a clean directory
4. Optionally, the LLM post-processes the transcript (clean up, restructure, extract ideas)

The output is a named directory with your audio and markdown:

```
your-project/
  quarterly-review-action-items/
    recording.mpeg
    transcript.md
    transcript_clean.md
```

Multiple audio files that belong to the same topic get merged into a single transcript automatically.

## Pluggable post-processing

The real power is in what happens **after** transcription. Voxtract ships with a `clean_transcript` MCP prompt, but the architecture is designed to be **forked and customized**.

The prompt templates live in a single file — `src/mistral_voice_mcp/prompts.py` — and each one is just a function that returns a message list. Want to turn Voxtract into a meeting assistant? A lecture note-taker? A medical dictation tool? **Write your own prompt function, register it in `server.py`, done.** Your LLM handles the rest.

```python
# src/mistral_voice_mcp/prompts.py — add your own

def meeting_notes_messages(transcript: str) -> list[dict]:
    return [{"role": "user", "content": f"Extract action items from:\n{transcript}"}]
```

```python
# src/mistral_voice_mcp/server.py — register it

@server.prompt(name="meeting_notes", description="Extract action items and decisions")
def meeting_notes(transcript: str) -> list[dict]:
    return prompts.meeting_notes_messages(transcript)
```

Some ideas for what you could build:

- **Meeting assistant** → Extract decisions, action items, attendees
- **Lecture notes** → Structured study material with key concepts
- **Lab notebook** → Extract methods, observations, technical terms
- **Interview processor** → Q&A format with key quotes highlighted
- **Brainstorm organizer** → Turn scattered spoken ideas into coherent proposals

Since the LLM you already use is the post-processing engine, each new use case costs you **zero extra** — just a new prompt function.

## Tools & Prompts

### Tools

| Tool | Description |
|------|-------------|
| `mistral_set_workdir` | Set the work directory. Creates `inbox/` and `.staging/` |
| `mistral_get_workdir` | Show current workdir path and status counts |
| `mistral_set_language` | Set transcription language (e.g. `italian`, `fr`). Non-English auto-disables timestamps |
| `mistral_get_language` | Show current language and timestamp status |
| `mistral_set_context_bias` | Set domain-specific terms for transcription accuracy (max 100) |
| `mistral_get_context_bias` | Show configured bias terms |
| `mistral_clear_context_bias` | Clear all bias terms |
| `mistral_list_inbox` | List audio files waiting in `inbox/` |
| `mistral_create_session` | Stage inbox files into a transcription session |
| `mistral_list_sessions` | List active staging sessions and their status |
| `mistral_transcribe` | Transcribe all audio files in a session (with diarization) |
| `mistral_read_staging_transcript` | Read merged transcript from staging before naming |
| `mistral_finalize` | Finalize session into a named directory with `transcript.md` |
| `mistral_list_recordings` | List all finalized recording directories |
| `mistral_read_transcript` | Read transcript (raw or clean) from a recording |
| `mistral_save_processed` | Save post-processed text as `transcript_clean.md` |

### Prompts

| Prompt | Description |
|--------|-------------|
| `clean_transcript` | Remove filler words, fix grammar and logic flow, restructure for clarity while preserving meaning and speaker headings |

## Use it everywhere

The idea is simple: add Voxtract to any repository where you work. Got an idea while walking? Record it, drop the file, and get structured documentation — right next to your code, notes, or research.

## Installation

No cloning, no setup. Requires [uv](https://docs.astral.sh/uv/) and a [Mistral API key](https://console.mistral.ai/) (pay-per-use, no subscription).

<details>
<summary><strong>Claude Code (CLI)</strong></summary>

```bash
# Available in all your projects
claude mcp add --scope user --env MISTRAL_API_KEY="your-key-here" voxtract -- uvx --from git+https://github.com/hitesit/voxtract voxtract
```

One command. Done. Use `--scope project` instead to limit it to the current project only.

</details>

<details>
<summary><strong>Any MCP client (manual JSON config)</strong></summary>

Add this to your MCP config (`.mcp.json` for Claude Code, `claude_desktop_config.json` for Claude Desktop, or equivalent for Cursor, Windsurf, etc.):

```json
{
  "mcpServers": {
    "voxtract": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "git+https://github.com/hitesit/voxtract", "voxtract"],
      "env": {
        "MISTRAL_API_KEY": "your-key-here"
      }
    }
  }
}
```

</details>

`uvx` downloads, installs, and runs Voxtract in an isolated environment automatically.

## Quick start

Once the MCP server is connected to your agent:

```
You: Set the work directory to /path/to/my/project
     I dropped a voice memo in the inbox, transcribe it in Italian and clean it up
```

The agent will:
1. Set the workdir, list the inbox
2. Create a staging session, transcribe the audio
3. Read the transcript, suggest a directory name
4. Finalize into a named folder with `transcript.md`
5. Clean and restructure into `transcript_clean.md`

All in one conversation, using tools you can see and control.

## Supported formats

`.flac`, `.m4a`, `.mp3`, `.mpeg`, `.ogg`, `.wav` — up to 1 GB, up to 3 hours per file.

## Supported languages

English, Chinese, Hindi, Spanish, Arabic, French, Portuguese, Russian, German, Japanese, Korean, Italian, Dutch.

## Cost

Mistral transcription: **$0.003 per minute** of audio. A 1-hour meeting costs ~$0.18. Post-processing is handled by the LLM you already use — no additional cost.

## License

MIT
