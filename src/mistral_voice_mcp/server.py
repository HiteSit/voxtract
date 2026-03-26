"""MCP server for audio transcription using Mistral Voxtral Mini Transcribe 2."""

import os
from pathlib import Path

from fastmcp import FastMCP, Context

from mistralai.client import Mistral

from mistral_voice_mcp import prompts
from mistral_voice_mcp.staging import (
    StagingSession,
    create_session,
    get_session,
    list_sessions,
)
from mistral_voice_mcp.transcriber import transcribe, MODEL_ID
from mistral_voice_mcp.workdir import AUDIO_EXTENSIONS, WorkDirectory

server = FastMCP(
    name="mistral-voice",
    instructions=(
        "Audio transcription server using Mistral Voxtral Mini Transcribe 2. "
        "Set a work directory first with mistral_set_workdir, then place audio "
        "files in the inbox/ subdirectory. Supported formats: "
        f"{', '.join(sorted(AUDIO_EXTENSIONS))}."
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_workdir(ctx: Context) -> WorkDirectory:
    """Retrieve WorkDirectory from session state."""
    path = await ctx.get_state("workdir_path")
    if path is None:
        raise ValueError(
            "No work directory set. Use mistral_set_workdir first."
        )
    return WorkDirectory(path)


def _get_client() -> Mistral:
    """Create Mistral client from environment variable."""
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable is not set.")
    return Mistral(api_key=api_key)


def _parse_bias_terms(terms: str) -> list[str]:
    """Parse context bias terms from comma or newline separated text."""
    raw = terms.replace(",", "\n")
    return [t.strip() for t in raw.splitlines() if t.strip()]


# Mapping of accepted language names/aliases → ISO 639-1 code
SUPPORTED_LANGUAGES: dict[str, str] = {
    "english": "en",
    "en": "en",
    "chinese": "zh",
    "zh": "zh",
    "mandarin": "zh",
    "hindi": "hi",
    "hi": "hi",
    "spanish": "es",
    "es": "es",
    "arabic": "ar",
    "ar": "ar",
    "french": "fr",
    "fr": "fr",
    "portuguese": "pt",
    "pt": "pt",
    "russian": "ru",
    "ru": "ru",
    "german": "de",
    "de": "de",
    "japanese": "ja",
    "ja": "ja",
    "korean": "ko",
    "ko": "ko",
    "italian": "it",
    "it": "it",
    "dutch": "nl",
    "nl": "nl",
}

# Default language setting
DEFAULT_LANGUAGE = "en"


async def _get_language(ctx: Context) -> str:
    """Retrieve the current language code from session state."""
    lang = await ctx.get_state("language")
    return lang or DEFAULT_LANGUAGE


# ---------------------------------------------------------------------------
# Tools: Work Directory
# ---------------------------------------------------------------------------

@server.tool(
    name="mistral_set_workdir",
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def set_workdir(path: str, ctx: Context) -> str:
    """Set the work directory for transcription. Creates inbox/ and .staging/ subdirectories if needed."""
    resolved = Path(path).expanduser().resolve()
    wd = WorkDirectory(resolved)
    await ctx.set_state("workdir_path", str(wd.root))
    s = wd.status()
    return (
        f"Work directory set to: {wd.root}\n"
        f"  inbox/      : {s['inbox']} audio files\n"
        f"  staging     : {s['staging']} sessions\n"
        f"  recordings  : {s['recordings']} finalized"
    )


@server.tool(
    name="mistral_get_workdir",
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def get_workdir(ctx: Context) -> str:
    """Show current work directory path and status."""
    wd = await _get_workdir(ctx)
    s = wd.status()
    bias = wd.load_context_bias()
    return (
        f"Work directory: {wd.root}\n"
        f"  inbox/      : {s['inbox']} audio files\n"
        f"  staging     : {s['staging']} sessions\n"
        f"  recordings  : {s['recordings']} finalized\n"
        f"  context bias: {len(bias)} terms"
    )


# ---------------------------------------------------------------------------
# Tools: Context Bias
# ---------------------------------------------------------------------------

@server.tool(
    name="mistral_set_context_bias",
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def set_context_bias(terms: str, ctx: Context) -> str:
    """Set context bias terms for better transcription accuracy.

    Accepts comma-separated or newline-separated terms (max 100).
    If 'terms' is a file path relative to the work directory, reads terms from that file.
    """
    wd = await _get_workdir(ctx)

    # Check if terms looks like a file path
    candidate = wd.root / terms
    if candidate.is_file():
        raw = candidate.read_text()
        parsed = _parse_bias_terms(raw)
    else:
        parsed = _parse_bias_terms(terms)

    wd.save_context_bias(parsed)
    return f"Context bias set: {len(parsed)} terms\n" + "\n".join(
        f"  - {t}" for t in parsed
    )


@server.tool(
    name="mistral_get_context_bias",
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def get_context_bias(ctx: Context) -> str:
    """Show currently configured context bias terms."""
    wd = await _get_workdir(ctx)
    terms = wd.load_context_bias()
    if not terms:
        return "No context bias terms configured."
    return f"Context bias ({len(terms)} terms):\n" + "\n".join(
        f"  - {t}" for t in terms
    )


@server.tool(
    name="mistral_clear_context_bias",
    annotations={
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def clear_context_bias(ctx: Context) -> str:
    """Clear all context bias terms."""
    wd = await _get_workdir(ctx)
    wd.clear_context_bias()
    return "Context bias cleared."


# ---------------------------------------------------------------------------
# Tools: Language
# ---------------------------------------------------------------------------

@server.tool(
    name="mistral_set_language",
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def set_language(language: str, ctx: Context) -> str:
    """Set the transcription language.

    Default is English with timestamps enabled.
    Setting a non-English language automatically disables timestamps
    (Mistral API constraint: language and timestamp_granularities are mutually exclusive).

    Args:
        language: Language name (e.g. 'italian', 'french') or ISO code (e.g. 'it', 'fr').
    """
    key = language.strip().lower()
    if key not in SUPPORTED_LANGUAGES:
        available = ", ".join(
            sorted({v: k for k, v in SUPPORTED_LANGUAGES.items() if len(k) > 2})
        )
        return f"Unsupported language: '{language}'. Available: {available}"

    code = SUPPORTED_LANGUAGES[key]
    await ctx.set_state("language", code)

    if code == DEFAULT_LANGUAGE:
        return f"Language set to: English (en). Timestamps: enabled."
    return (
        f"Language set to: {key} ({code}). Timestamps: disabled "
        f"(mutually exclusive with language setting)."
    )


@server.tool(
    name="mistral_get_language",
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def get_language(ctx: Context) -> str:
    """Show current transcription language and whether timestamps are active."""
    code = await _get_language(ctx)
    # Reverse-lookup the full name
    name = next(
        (k for k, v in SUPPORTED_LANGUAGES.items() if v == code and len(k) > 2),
        code,
    )
    timestamps = "enabled" if code == DEFAULT_LANGUAGE else "disabled"
    return f"Language: {name} ({code}). Timestamps: {timestamps}."


# ---------------------------------------------------------------------------
# Tools: Inbox
# ---------------------------------------------------------------------------

@server.tool(
    name="mistral_list_inbox",
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def list_inbox(ctx: Context) -> str:
    """List audio files in inbox/ waiting to be transcribed."""
    wd = await _get_workdir(ctx)
    files = wd.scan_inbox()
    if not files:
        return "No audio files found in inbox/."

    lines = [f"Found {len(files)} audio files in inbox/:\n"]
    for f in files:
        rel = f.relative_to(wd.inbox_dir)
        size_kb = f.stat().st_size / 1024
        lines.append(f"  {rel} ({size_kb:.1f} KB)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tools: Staging Sessions
# ---------------------------------------------------------------------------

@server.tool(
    name="mistral_create_session",
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def create_session_tool(
    ctx: Context, filenames: list[str] | None = None
) -> str:
    """Create a staging session from audio files in inbox/.

    Copies the specified files (or all inbox files) into a staging session
    for transcription.

    Args:
        filenames: List of filenames in inbox/ to include. None means all files.
    """
    wd = await _get_workdir(ctx)

    if filenames:
        audio_paths = []
        for name in filenames:
            path = wd.inbox_dir / name
            if not path.exists():
                return f"File not found in inbox: {name}"
            audio_paths.append(path)
    else:
        audio_paths = wd.scan_inbox()
        if not audio_paths:
            return "No audio files in inbox/."

    session = create_session(wd.root, audio_paths)
    file_list = "\n".join(f"  - {p.name}" for p in session.audio_files)
    return (
        f"Session created: {session.session_id}\n"
        f"  Files: {len(session.audio_files)}\n"
        f"{file_list}"
    )


@server.tool(
    name="mistral_list_sessions",
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def list_sessions_tool(ctx: Context) -> str:
    """List active staging sessions."""
    wd = await _get_workdir(ctx)
    sessions = list_sessions(wd.root)
    if not sessions:
        return "No active staging sessions."

    lines = [f"Found {len(sessions)} staging sessions:\n"]
    for s in sessions:
        n_audio = len(s.audio_files)
        transcribed = s.is_fully_transcribed()
        status = "transcribed" if transcribed else "pending"
        lines.append(f"  {s.session_id} — {n_audio} files [{status}]")
    return "\n".join(lines)


@server.tool(
    name="mistral_read_staging_transcript",
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def read_staging_transcript(session_id: str, ctx: Context) -> str:
    """Read the merged transcript from a staging session.

    Use this to review the transcript before finalizing with a name.

    Args:
        session_id: The staging session ID.
    """
    wd = await _get_workdir(ctx)
    session = get_session(wd.root, session_id)

    if not session.is_fully_transcribed():
        return f"Session {session_id} is not fully transcribed yet."

    merged = session.merge_transcripts()
    if not merged:
        return f"No transcripts found in session {session_id}."
    return merged


# ---------------------------------------------------------------------------
# Tools: Transcription
# ---------------------------------------------------------------------------

@server.tool(
    name="mistral_transcribe",
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def transcribe_session(
    session_id: str,
    ctx: Context,
    diarize: bool = True,
    timestamp_granularity: str = "segment",
) -> str:
    """Transcribe all audio files in a staging session.

    Uses the session language set via mistral_set_language (default: English).

    Args:
        session_id: The staging session ID.
        diarize: Enable speaker identification.
        timestamp_granularity: 'segment' or 'word' level timestamps (ignored for non-English).
    """
    wd = await _get_workdir(ctx)
    client = _get_client()
    session = get_session(wd.root, session_id)

    files = session.audio_files
    if not files:
        return f"No audio files in session {session_id}."

    bias = wd.load_context_bias() or None
    lang_code = await _get_language(ctx)
    language = None if lang_code == DEFAULT_LANGUAGE else lang_code

    total = len(files)
    results: list[str] = []
    errors: list[str] = []

    for i, audio in enumerate(files):
        await ctx.report_progress(i, total)
        await ctx.info(f"Transcribing [{i + 1}/{total}] {audio.name} (lang={lang_code})")

        try:
            await transcribe(
                client,
                audio,
                session.session_dir,
                diarize=diarize,
                timestamp_granularity=timestamp_granularity,
                context_bias=bias,
                language=language,
            )
            results.append(audio.name)
        except Exception as e:
            errors.append(f"{audio.name}: {e}")

    await ctx.report_progress(total, total)

    lines = [
        f"Transcription complete: {len(results)}/{total} files",
        f"  Session: {session_id}",
        f"  Language: {lang_code}",
    ]
    if results:
        lines.append("\nCompleted:")
        lines.extend(f"  - {r}" for r in results)
    if errors:
        lines.append("\nErrors:")
        lines.extend(f"  - {e}" for e in errors)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tools: Finalize
# ---------------------------------------------------------------------------

@server.tool(
    name="mistral_finalize",
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def finalize_session(
    session_id: str, name: str, ctx: Context
) -> str:
    """Finalize a staging session into a named recording directory.

    Creates a directory under workdir with the given name (slugified),
    moves audio files into it, writes the merged transcript.md,
    and cleans up the staging session and inbox originals.

    Args:
        session_id: The staging session ID.
        name: A descriptive name for the recording (will be slugified).
    """
    wd = await _get_workdir(ctx)
    session = get_session(wd.root, session_id)

    if not session.is_fully_transcribed():
        return f"Session {session_id} is not fully transcribed yet."

    # Remember inbox files to clean up after finalize
    inbox_names = {p.name for p in session.audio_files}

    try:
        final_dir = session.finalize(name)
    except ValueError as e:
        return str(e)

    # Clean up inbox originals
    for inbox_file in wd.inbox_dir.iterdir():
        if inbox_file.name in inbox_names:
            inbox_file.unlink()

    contents = sorted(p.name for p in final_dir.iterdir())
    file_list = "\n".join(f"  - {c}" for c in contents)
    return (
        f"Finalized: {final_dir.name}/\n"
        f"  Path: {final_dir}\n"
        f"  Contents:\n{file_list}"
    )


# ---------------------------------------------------------------------------
# Tools: Recordings
# ---------------------------------------------------------------------------

@server.tool(
    name="mistral_list_recordings",
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def list_recordings(ctx: Context) -> str:
    """List finalized recording directories."""
    wd = await _get_workdir(ctx)
    recordings = wd.scan_recordings()
    if not recordings:
        return "No finalized recordings."

    lines = [f"Found {len(recordings)} recordings:\n"]
    for d in recordings:
        audio = [p.name for p in d.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS]
        has_transcript = (d / "transcript.md").exists()
        has_clean = (d / "transcript_clean.md").exists()
        status = []
        if has_transcript:
            status.append("transcript")
        if has_clean:
            status.append("clean")
        lines.append(
            f"  {d.name}/ — {len(audio)} audio, [{', '.join(status)}]"
        )
    return "\n".join(lines)


@server.tool(
    name="mistral_read_transcript",
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def read_transcript(
    recording: str, ctx: Context, clean: bool = False
) -> str:
    """Read a transcript from a finalized recording.

    Args:
        recording: Recording directory name (e.g. 'orca-software-dft').
        clean: If True, read transcript_clean.md instead of transcript.md.
    """
    wd = await _get_workdir(ctx)
    rec_dir = wd.root / recording
    if not rec_dir.is_dir():
        return f"Recording not found: {recording}"

    filename = "transcript_clean.md" if clean else "transcript.md"
    path = rec_dir / filename
    if not path.exists():
        return f"File not found: {recording}/{filename}"

    return path.read_text()


# ---------------------------------------------------------------------------
# Tools: Post-processing
# ---------------------------------------------------------------------------

@server.tool(
    name="mistral_save_processed",
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def save_processed(
    recording: str, content: str, ctx: Context
) -> str:
    """Save a cleaned transcript into a finalized recording directory.

    Writes the content as transcript_clean.md.

    Args:
        recording: Recording directory name (e.g. 'orca-software-dft').
        content: The cleaned markdown content to save.
    """
    wd = await _get_workdir(ctx)
    rec_dir = wd.root / recording
    if not rec_dir.is_dir():
        return f"Recording not found: {recording}"

    clean_path = rec_dir / "transcript_clean.md"
    clean_path.write_text(content)

    return f"Saved cleaned transcript: {clean_path}"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

@server.prompt(
    name="clean_transcript",
    description=(
        "Clean a raw transcript: remove filler words, fix grammar and logic flow, "
        "restructure for clarity while preserving meaning and speaker headings. "
        "After the LLM produces the cleaned text, save it with mistral_save_processed."
    ),
)
def clean_transcript(transcript: str) -> list[dict]:
    return prompts.clean_transcript_messages(transcript)


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@server.resource(
    "transcription://supported-formats",
    name="supported_formats",
    description="Supported audio formats, limits, and model information.",
    mime_type="application/json",
)
def supported_formats() -> dict:
    return {
        "formats": sorted(AUDIO_EXTENSIONS),
        "max_file_size": "1 GB",
        "max_duration": "3 hours",
        "model": MODEL_ID,
        "pricing": "$0.003 per minute of audio",
        "supported_languages": [
            "English", "Chinese", "Hindi", "Spanish", "Arabic",
            "French", "Portuguese", "Russian", "German", "Japanese",
            "Korean", "Italian", "Dutch",
        ],
        "features": {
            "diarization": "Speaker identification (diarize=True)",
            "timestamps": "Segment-level or word-level",
            "context_bias": "Up to 100 terms for spelling guidance",
        },
        "notes": [
            "timestamp_granularities and language are mutually exclusive",
            "Context bias is optimized for English, experimental for others",
        ],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    server.run()


if __name__ == "__main__":
    main()
