"""Shared test fixtures."""

import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def example_audio() -> Path:
    """Path to the real Example.mpeg test fixture."""
    p = FIXTURES_DIR / "Example.mpeg"
    assert p.exists(), f"Test fixture not found: {p}"
    return p


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """Create a temporary work directory with input/ and output/ subdirs."""
    (tmp_path / "input").mkdir()
    (tmp_path / "output").mkdir()
    return tmp_path


@pytest.fixture
def workdir_with_audio(workdir: Path, example_audio: Path) -> Path:
    """Work directory with Example.mpeg copied into input/."""
    shutil.copy2(example_audio, workdir / "input" / "Example.mpeg")
    return workdir


@pytest.fixture
def mock_mistral_client() -> MagicMock:
    """Mock Mistral client with a pre-configured transcription response."""
    client = MagicMock()

    mock_response = MagicMock()
    mock_response.text = "Transcribed text content."
    mock_response.model = "voxtral-mini-2602"
    mock_response.language = None
    mock_response.segments = [
        MagicMock(
            text="Transcribed text content.",
            start=0.0,
            end=5.0,
            type="transcription_segment",
            speaker_id="speaker_1",
        )
    ]
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 50
    mock_response.usage.total_tokens = 60
    mock_response.usage.prompt_audio_seconds = 30
    mock_response.model_dump.return_value = {
        "model": "voxtral-mini-2602",
        "text": "Transcribed text content.",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 50,
            "total_tokens": 60,
            "prompt_audio_seconds": 30,
        },
        "language": None,
        "segments": [
            {
                "text": "Transcribed text content.",
                "start": 0.0,
                "end": 5.0,
                "type": "transcription_segment",
                "speaker_id": "speaker_1",
            }
        ],
    }

    client.audio.transcriptions.complete_async = AsyncMock(
        return_value=mock_response
    )
    return client
