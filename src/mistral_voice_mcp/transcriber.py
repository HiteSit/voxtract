"""Mistral audio transcription wrapper."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from mistralai.client import Mistral

from mistral_voice_mcp.workdir import WorkDirectory

MODEL_ID: str = "voxtral-mini-2602"


@dataclass
class TranscriptionResult:
    """Container for transcription output."""

    text: str
    segments: list[dict] = field(default_factory=list)
    language: str | None = None
    usage: dict = field(default_factory=dict)
    model: str = MODEL_ID
    json_path: Path | None = None
    md_path: Path | None = None


def _segments_to_markdown(segments: list[dict]) -> str:
    """Convert segments into speaker-grouped markdown.

    Consecutive segments from the same speaker are concatenated under
    a single ``## Speaker N`` heading.
    """
    if not segments:
        return ""

    blocks: list[tuple[str, str]] = []  # (speaker_label, text)
    for seg in segments:
        speaker = seg.get("speaker_id") or "unknown"
        text = seg.get("text", "").strip()
        if not text:
            continue
        if blocks and blocks[-1][0] == speaker:
            blocks[-1] = (speaker, blocks[-1][1] + " " + text)
        else:
            blocks.append((speaker, text))

    lines: list[str] = []
    for speaker, text in blocks:
        label = speaker.replace("_", " ").title()
        lines.append(f"## {label}\n")
        lines.append(f"{text}\n")

    return "\n".join(lines)


async def transcribe(
    client: Mistral,
    workdir: WorkDirectory,
    input_path: Path,
    *,
    diarize: bool = True,
    timestamp_granularity: str = "segment",
    context_bias: list[str] | None = None,
    language: str | None = None,
) -> TranscriptionResult:
    """Transcribe a single audio file and save results to output/.

    Args:
        client: Mistral API client.
        workdir: Work directory managing input/output paths.
        input_path: Path to audio file (must be within workdir.input_dir).
        diarize: Enable speaker diarization.
        timestamp_granularity: "segment" or "word" level timestamps.
        context_bias: List of terms to guide transcription spelling.
        language: Language code (mutually exclusive with timestamps).
    """
    input_path = Path(input_path).resolve()
    try:
        input_path.relative_to(workdir.input_dir)
    except ValueError:
        raise ValueError(
            f"File {input_path} is not within the input directory {workdir.input_dir}"
        )

    kwargs: dict = {
        "model": MODEL_ID,
        "diarize": diarize,
    }

    # timestamp_granularities and language are mutually exclusive
    if language:
        kwargs["language"] = language
    else:
        kwargs["timestamp_granularities"] = [timestamp_granularity]

    if context_bias:
        kwargs["context_bias"] = context_bias

    with open(input_path, "rb") as f:
        kwargs["file"] = {"content": f, "file_name": input_path.name}
        response = await client.audio.transcriptions.complete_async(**kwargs)

    # Save JSON output
    json_path = workdir.get_output_path(input_path, suffix=".json")
    json_path.write_text(
        json.dumps(response.model_dump(), indent=2, default=str)
    )

    # Build structured segments list
    segments = [
        {
            "text": seg.text,
            "start": seg.start,
            "end": seg.end,
            "speaker_id": getattr(seg, "speaker_id", None),
        }
        for seg in (response.segments or [])
    ]

    # Save markdown with speaker-grouped conversation format
    md_path = workdir.get_output_path(input_path, suffix=".md")
    md_path.write_text(_segments_to_markdown(segments))

    return TranscriptionResult(
        text=response.text,
        segments=segments,
        language=response.language,
        usage=response.usage.model_dump() if hasattr(response.usage, "model_dump") else {},
        model=response.model,
        json_path=json_path,
        md_path=md_path,
    )
