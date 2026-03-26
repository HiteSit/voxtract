"""MCP server for audio transcription using Mistral Voxtral Mini Transcribe 2."""

import os
from pathlib import Path

from fastmcp import FastMCP, Context

from mistralai.client import Mistral

from mistral_voice_mcp import prompts
from mistral_voice_mcp.transcriber import transcribe, MODEL_ID
from mistral_voice_mcp.workdir import AUDIO_EXTENSIONS, WorkDirectory

server = FastMCP(
    name="mistral-voice",
    instructions=(
        "Audio transcription server using Mistral Voxtral Mini Transcribe 2. "
        "Set a work directory first with mistral_set_workdir, then place audio "
        "files in the input/ subdirectory. Supported formats: "
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
    """Set the work directory for transcription. Creates input/ and output/ subdirectories if needed."""
    resolved = Path(path).expanduser().resolve()
    wd = WorkDirectory(resolved)
    await ctx.set_state("workdir_path", str(wd.root))
    s = wd.status()
    return (
        f"Work directory set to: {wd.root}\n"
        f"  input/  : {s['total_inputs']} audio files\n"
        f"  output/ : {s['transcribed']} transcriptions\n"
        f"  pending : {s['pending']} files to transcribe"
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
        f"  input/  : {s['total_inputs']} audio files\n"
        f"  output/ : {s['transcribed']} transcriptions\n"
        f"  pending : {s['pending']} files to transcribe\n"
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
# Tools: Input / Output Listing
# ---------------------------------------------------------------------------

@server.tool(
    name="mistral_list_inputs",
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def list_inputs(ctx: Context) -> str:
    """List audio files in input/ (recursive), showing transcription status."""
    wd = await _get_workdir(ctx)
    files = wd.scan_inputs()
    if not files:
        return "No audio files found in input/."

    lines = [f"Found {len(files)} audio files:\n"]
    for f in files:
        rel = f.relative_to(wd.input_dir)
        status = "done" if wd.is_transcribed(f) else "pending"
        lines.append(f"  [{status}] {rel}")
    return "\n".join(lines)


@server.tool(
    name="mistral_list_transcriptions",
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def list_transcriptions(ctx: Context) -> str:
    """List completed transcriptions in output/."""
    wd = await _get_workdir(ctx)
    json_files = sorted(wd.output_dir.rglob("*.json"))
    if not json_files:
        return "No transcriptions found in output/."

    lines = [f"Found {len(json_files)} transcriptions:\n"]
    for f in json_files:
        rel = f.relative_to(wd.output_dir)
        size_kb = f.stat().st_size / 1024
        md_exists = f.with_suffix(".md").exists()
        lines.append(
            f"  {rel} ({size_kb:.1f} KB)"
            + (" [+md]" if md_exists else "")
        )
    return "\n".join(lines)


@server.tool(
    name="mistral_read_transcription",
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def read_transcription(
    filename: str, ctx: Context, format: str = "md"
) -> str:
    """Read a specific transcription result.

    Args:
        filename: Filename relative to output/ (e.g. 'Example.md' or 'batch/call.json').
        format: 'md' for markdown transcript or 'json' for full structured output.
    """
    wd = await _get_workdir(ctx)

    # Resolve the file, allowing user to omit the extension
    candidate = wd.output_dir / filename
    if not candidate.exists():
        candidate = wd.output_dir / Path(filename).with_suffix(
            f".{format}"
        )
    if not candidate.exists():
        return f"Transcription not found: {filename}"

    return candidate.read_text()


# ---------------------------------------------------------------------------
# Tools: Transcription
# ---------------------------------------------------------------------------

@server.tool(
    name="mistral_transcribe_file",
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def transcribe_file(
    filename: str,
    ctx: Context,
    diarize: bool = True,
    timestamp_granularity: str = "segment",
) -> str:
    """Transcribe a single audio file from input/.

    Uses the session language set via mistral_set_language (default: English).
    When language is non-English, timestamps are automatically disabled.

    Args:
        filename: Path relative to input/ (e.g. 'Example.mpeg' or 'batch/call.mp3').
        diarize: Enable speaker identification.
        timestamp_granularity: 'segment' or 'word' level timestamps (ignored for non-English).
    """
    wd = await _get_workdir(ctx)
    client = _get_client()

    input_path = wd.input_dir / filename
    if not input_path.exists():
        return f"File not found: {input_path}"

    bias = wd.load_context_bias() or None
    lang_code = await _get_language(ctx)
    # English uses timestamps; non-English uses language param instead
    language = None if lang_code == DEFAULT_LANGUAGE else lang_code

    await ctx.info(f"Transcribing {filename} (lang={lang_code})...")
    result = await transcribe(
        client,
        wd,
        input_path,
        diarize=diarize,
        timestamp_granularity=timestamp_granularity,
        context_bias=bias,
        language=language,
    )

    n_segments = len(result.segments)
    speakers = {s.get("speaker_id") for s in result.segments if s.get("speaker_id")}

    return (
        f"Transcription complete: {filename}\n"
        f"  Model: {result.model}\n"
        f"  Language: {lang_code}\n"
        f"  Text length: {len(result.text)} chars\n"
        f"  Segments: {n_segments}\n"
        f"  Speakers: {len(speakers) if speakers else 'N/A'}\n"
        f"  Timestamps: {'enabled' if language is None else 'disabled'}\n"
        f"  Saved: {result.json_path}\n"
        f"          {result.md_path}"
    )


@server.tool(
    name="mistral_transcribe_batch",
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def transcribe_batch(
    ctx: Context,
    subdirectory: str | None = None,
    force: bool = False,
    diarize: bool = True,
    timestamp_granularity: str = "segment",
) -> str:
    """Transcribe all untranscribed audio files in input/ (or a subdirectory).

    Uses the session language set via mistral_set_language (default: English).

    Args:
        subdirectory: Optional subdirectory within input/ to scope the batch.
        force: Re-transcribe files that already have output.
        diarize: Enable speaker identification.
        timestamp_granularity: 'segment' or 'word' level timestamps (ignored for non-English).
    """
    wd = await _get_workdir(ctx)
    client = _get_client()

    if force:
        files = wd.scan_inputs()
    else:
        files = wd.pending_files()

    if subdirectory:
        sub_path = wd.input_dir / subdirectory
        files = [f for f in files if str(f).startswith(str(sub_path))]

    if not files:
        return "No files to transcribe."

    bias = wd.load_context_bias() or None
    lang_code = await _get_language(ctx)
    language = None if lang_code == DEFAULT_LANGUAGE else lang_code
    total = len(files)
    results: list[str] = []
    errors: list[str] = []

    for i, f in enumerate(files):
        rel = f.relative_to(wd.input_dir)
        await ctx.report_progress(i, total)
        await ctx.info(f"Transcribing [{i + 1}/{total}] {rel} (lang={lang_code})")

        try:
            await transcribe(
                client,
                wd,
                f,
                diarize=diarize,
                timestamp_granularity=timestamp_granularity,
                context_bias=bias,
                language=language,
            )
            results.append(str(rel))
        except Exception as e:
            errors.append(f"{rel}: {e}")

    await ctx.report_progress(total, total)

    lines = [f"Batch transcription complete: {len(results)}/{total} succeeded"]
    if results:
        lines.append("\nCompleted:")
        lines.extend(f"  - {r}" for r in results)
    if errors:
        lines.append("\nErrors:")
        lines.extend(f"  - {e}" for e in errors)

    return "\n".join(lines)


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
    filename: str, content: str, ctx: Context
) -> str:
    """Save a processed (cleaned) transcript to output/.

    Writes the content as a *_clean.md file alongside the original transcription.

    Args:
        filename: Original transcription filename relative to output/
                  (e.g. 'Example.md' or 'Example'). The '_clean' suffix is added automatically.
        content: The cleaned markdown content to save.
    """
    wd = await _get_workdir(ctx)

    # Normalise: strip known suffixes so we always build from the stem
    name = Path(filename).with_suffix("").name
    if name.endswith("_clean"):
        name = name[: -len("_clean")]

    # Preserve subdirectory structure
    parent = Path(filename).parent
    clean_path = wd.output_dir / parent / f"{name}_clean.md"
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean_path.write_text(content)

    return f"Saved processed transcript: {clean_path}"


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
