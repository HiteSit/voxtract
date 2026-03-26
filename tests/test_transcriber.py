"""Tests for the transcriber module."""

import os
import re
from pathlib import Path

import pytest

from mistral_voice_mcp.transcriber import MODEL_ID, _segments_to_markdown, transcribe


@pytest.fixture
def audio_and_outdir(tmp_path: Path) -> tuple[Path, Path]:
    """A dummy audio file and an output directory."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"\x00" * 100)
    out = tmp_path / "output"
    out.mkdir()
    return audio, out


class TestTranscribe:
    @pytest.mark.asyncio
    async def test_calls_api_with_correct_params(
        self, audio_and_outdir, mock_mistral_client
    ):
        audio, out = audio_and_outdir
        await transcribe(mock_mistral_client, audio, out)

        call_kwargs = mock_mistral_client.audio.transcriptions.complete_async.call_args
        assert call_kwargs.kwargs["model"] == MODEL_ID
        assert call_kwargs.kwargs["diarize"] is True
        assert call_kwargs.kwargs["timestamp_granularities"] == ["segment"]

    @pytest.mark.asyncio
    async def test_saves_json_output(self, audio_and_outdir, mock_mistral_client):
        audio, out = audio_and_outdir
        result = await transcribe(mock_mistral_client, audio, out)

        assert result.json_path is not None
        assert result.json_path.exists()
        assert result.json_path.suffix == ".json"
        assert result.json_path.parent == out

    @pytest.mark.asyncio
    async def test_saves_md_output(self, audio_and_outdir, mock_mistral_client):
        audio, out = audio_and_outdir
        result = await transcribe(mock_mistral_client, audio, out)

        assert result.md_path is not None
        assert result.md_path.exists()
        assert result.md_path.suffix == ".md"
        content = result.md_path.read_text()
        assert "## Speaker 1" in content
        assert "Transcribed text content." in content

    @pytest.mark.asyncio
    async def test_context_bias_forwarded(
        self, audio_and_outdir, mock_mistral_client
    ):
        audio, out = audio_and_outdir
        bias = ["ORCA", "DFT", "MCR"]
        await transcribe(mock_mistral_client, audio, out, context_bias=bias)

        call_kwargs = mock_mistral_client.audio.transcriptions.complete_async.call_args
        assert call_kwargs.kwargs["context_bias"] == bias

    @pytest.mark.asyncio
    async def test_language_with_diarize_keeps_timestamps(
        self, audio_and_outdir, mock_mistral_client
    ):
        """Diarization requires timestamps, so both are sent even with language."""
        audio, out = audio_and_outdir
        await transcribe(mock_mistral_client, audio, out, language="it", diarize=True)

        call_kwargs = mock_mistral_client.audio.transcriptions.complete_async.call_args
        assert call_kwargs.kwargs["language"] == "it"
        assert call_kwargs.kwargs["timestamp_granularities"] == ["segment"]
        assert call_kwargs.kwargs["diarize"] is True

    @pytest.mark.asyncio
    async def test_language_without_diarize_excludes_timestamps(
        self, audio_and_outdir, mock_mistral_client
    ):
        """Without diarization, language alone excludes timestamps."""
        audio, out = audio_and_outdir
        await transcribe(mock_mistral_client, audio, out, language="it", diarize=False)

        call_kwargs = mock_mistral_client.audio.transcriptions.complete_async.call_args
        assert call_kwargs.kwargs["language"] == "it"
        assert "timestamp_granularities" not in call_kwargs.kwargs

    @pytest.mark.asyncio
    async def test_creates_output_dir_if_missing(self, tmp_path, mock_mistral_client):
        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"\x00" * 100)
        out = tmp_path / "new" / "nested" / "dir"

        result = await transcribe(mock_mistral_client, audio, out)
        assert result.json_path.exists()
        assert result.md_path.exists()

    @pytest.mark.asyncio
    async def test_output_uses_audio_stem(self, audio_and_outdir, mock_mistral_client):
        audio, out = audio_and_outdir
        result = await transcribe(mock_mistral_client, audio, out)
        assert result.json_path.stem == "test"
        assert result.md_path.stem == "test"


class TestSegmentsToMarkdown:
    def test_empty_segments(self):
        assert _segments_to_markdown([]) == ""

    def test_single_speaker_concatenation(self):
        segments = [
            {"text": "Hello there.", "start": 0.0, "end": 2.0, "speaker_id": "speaker_1"},
            {"text": "How are you?", "start": 2.5, "end": 4.0, "speaker_id": "speaker_1"},
        ]
        md = _segments_to_markdown(segments)
        assert md.count("## Speaker 1") == 1
        assert "Hello there. How are you?" in md

    def test_multiple_speakers_alternate(self):
        segments = [
            {"text": "Hi.", "start": 0.0, "end": 1.0, "speaker_id": "speaker_1"},
            {"text": "Hey!", "start": 1.5, "end": 2.0, "speaker_id": "speaker_2"},
            {"text": "What's up?", "start": 2.5, "end": 3.0, "speaker_id": "speaker_1"},
        ]
        md = _segments_to_markdown(segments)
        assert md.count("## Speaker 1") == 2
        assert md.count("## Speaker 2") == 1
        headings = re.findall(r"## (Speaker \d)", md)
        assert headings == ["Speaker 1", "Speaker 2", "Speaker 1"]

    def test_speaker_returns_after_other(self):
        segments = [
            {"text": "A", "start": 0.0, "end": 1.0, "speaker_id": "speaker_1"},
            {"text": "B", "start": 1.0, "end": 2.0, "speaker_id": "speaker_1"},
            {"text": "C", "start": 2.0, "end": 3.0, "speaker_id": "speaker_2"},
            {"text": "D", "start": 3.0, "end": 4.0, "speaker_id": "speaker_1"},
        ]
        md = _segments_to_markdown(segments)
        assert "A B" in md
        assert "C" in md
        assert "D" in md

    def test_missing_speaker_id(self):
        segments = [
            {"text": "No speaker.", "start": 0.0, "end": 1.0},
        ]
        md = _segments_to_markdown(segments)
        assert "## Unknown" in md


@pytest.mark.integration
class TestTranscribeEndToEnd:
    """End-to-end test using real Mistral API and the Example.mpeg fixture.

    Run with: uv run pytest -m integration
    """

    @pytest.mark.asyncio
    async def test_transcribe_produces_valid_markdown(self, tmp_path: Path):
        from mistralai.client import Mistral

        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            pytest.skip("MISTRAL_API_KEY not set")

        fixture = Path(__file__).parent / "fixtures" / "Example.mpeg"
        if not fixture.exists():
            pytest.skip("Example.mpeg fixture not found")

        client = Mistral(api_key=api_key)
        out = tmp_path / "output"

        result = await transcribe(client, fixture, out)

        # JSON output exists
        assert result.json_path is not None
        assert result.json_path.exists()

        # Markdown output exists with valid structure
        assert result.md_path is not None
        assert result.md_path.exists()

        md_content = result.md_path.read_text()
        headings = re.findall(r"^## .+$", md_content, re.MULTILINE)
        assert len(headings) >= 1
        for h in headings:
            assert re.match(r"## Speaker \d+", h)

        # No .txt file produced
        assert not (out / "Example.txt").exists()
