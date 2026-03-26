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
    txt_path: Path | None = None


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

    # Save plain text output
    txt_path = workdir.get_output_path(input_path, suffix=".txt")
    txt_path.write_text(response.text)

    return TranscriptionResult(
        text=response.text,
        segments=[
            {
                "text": seg.text,
                "start": seg.start,
                "end": seg.end,
                "speaker_id": getattr(seg, "speaker_id", None),
            }
            for seg in (response.segments or [])
        ],
        language=response.language,
        usage=response.usage.model_dump() if hasattr(response.usage, "model_dump") else {},
        model=response.model,
        json_path=json_path,
        txt_path=txt_path,
    )
