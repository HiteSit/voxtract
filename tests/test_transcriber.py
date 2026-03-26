"""Tests for the transcriber module."""

import os
import re
from pathlib import Path

import pytest

from mistral_voice_mcp.transcriber import MODEL_ID, _segments_to_markdown, transcribe
from mistral_voice_mcp.workdir import WorkDirectory


@pytest.fixture
def wd_with_file(workdir: Path) -> tuple[WorkDirectory, Path]:
    """WorkDirectory with a dummy audio file in input/."""
    wd = WorkDirectory(workdir)
    audio = wd.input_dir / "test.mp3"
    audio.write_bytes(b"\x00" * 100)
    return wd, audio


class TestTranscribe:
    @pytest.mark.asyncio
    async def test_calls_api_with_correct_params(
        self, wd_with_file, mock_mistral_client
    ):
        wd, audio = wd_with_file
        await transcribe(mock_mistral_client, wd, audio)

        call_kwargs = mock_mistral_client.audio.transcriptions.complete_async.call_args
        assert call_kwargs.kwargs["model"] == MODEL_ID
        assert call_kwargs.kwargs["diarize"] is True
        assert call_kwargs.kwargs["timestamp_granularities"] == ["segment"]

    @pytest.mark.asyncio
    async def test_saves_json_output(self, wd_with_file, mock_mistral_client):
        wd, audio = wd_with_file
        result = await transcribe(mock_mistral_client, wd, audio)

        assert result.json_path is not None
        assert result.json_path.exists()
        assert result.json_path.suffix == ".json"

    @pytest.mark.asyncio
    async def test_saves_md_output(self, wd_with_file, mock_mistral_client):
        wd, audio = wd_with_file
        result = await transcribe(mock_mistral_client, wd, audio)

        assert result.md_path is not None
        assert result.md_path.exists()
        assert result.md_path.suffix == ".md"
        content = result.md_path.read_text()
        assert "## Speaker 1" in content
        assert "Transcribed text content." in content

    @pytest.mark.asyncio
    async def test_context_bias_forwarded(
        self, wd_with_file, mock_mistral_client
    ):
        wd, audio = wd_with_file
        bias = ["ORCA", "DFT", "MCR"]
        await transcribe(
            mock_mistral_client, wd, audio, context_bias=bias
        )

        call_kwargs = mock_mistral_client.audio.transcriptions.complete_async.call_args
        assert call_kwargs.kwargs["context_bias"] == bias

    @pytest.mark.asyncio
    async def test_language_with_diarize_keeps_timestamps(
        self, wd_with_file, mock_mistral_client
    ):
        """Diarization requires timestamps, so both are sent even with language."""
        wd, audio = wd_with_file
        await transcribe(
            mock_mistral_client, wd, audio, language="it", diarize=True
        )

        call_kwargs = mock_mistral_client.audio.transcriptions.complete_async.call_args
        assert call_kwargs.kwargs["language"] == "it"
        assert call_kwargs.kwargs["timestamp_granularities"] == ["segment"]
        assert call_kwargs.kwargs["diarize"] is True

    @pytest.mark.asyncio
    async def test_language_without_diarize_excludes_timestamps(
        self, wd_with_file, mock_mistral_client
    ):
        """Without diarization, language alone excludes timestamps."""
        wd, audio = wd_with_file
        await transcribe(
            mock_mistral_client, wd, audio, language="it", diarize=False
        )

        call_kwargs = mock_mistral_client.audio.transcriptions.complete_async.call_args
        assert call_kwargs.kwargs["language"] == "it"
        assert "timestamp_granularities" not in call_kwargs.kwargs

    @pytest.mark.asyncio
    async def test_rejects_file_outside_input_dir(
        self, workdir: Path, mock_mistral_client
    ):
        wd = WorkDirectory(workdir)
        outside_file = workdir / "not_in_input.mp3"
        outside_file.write_bytes(b"\x00")

        with pytest.raises(ValueError, match="not within the input directory"):
            await transcribe(mock_mistral_client, wd, outside_file)

    @pytest.mark.asyncio
    async def test_output_mirrors_subdirectory(
        self, workdir: Path, mock_mistral_client
    ):
        wd = WorkDirectory(workdir)
        sub = wd.input_dir / "batch" / "day1"
        sub.mkdir(parents=True)
        audio = sub / "call.mp3"
        audio.write_bytes(b"\x00")

        result = await transcribe(mock_mistral_client, wd, audio)
        assert "batch" in str(result.json_path)
        assert "day1" in str(result.json_path)


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
    async def test_transcribe_produces_valid_markdown(
        self, workdir_with_audio: Path
    ):
        from mistralai.client import Mistral

        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            pytest.skip("MISTRAL_API_KEY not set")

        client = Mistral(api_key=api_key)
        wd = WorkDirectory(workdir_with_audio)
        audio = wd.input_dir / "Example.mpeg"

        result = await transcribe(client, wd, audio)

        # JSON output exists and is valid
        assert result.json_path is not None
        assert result.json_path.exists()
        assert result.json_path.suffix == ".json"

        # Markdown output exists
        assert result.md_path is not None
        assert result.md_path.exists()
        assert result.md_path.suffix == ".md"

        md_content = result.md_path.read_text()

        # Has at least one speaker heading
        headings = re.findall(r"^## .+$", md_content, re.MULTILINE)
        assert len(headings) >= 1, "Markdown must have at least one speaker heading"

        # Every heading matches the expected format
        for h in headings:
            assert re.match(r"## Speaker \d+", h), f"Unexpected heading format: {h}"

        # Content is non-empty between headings
        blocks = re.split(r"^## .+$", md_content, flags=re.MULTILINE)
        non_empty_blocks = [b.strip() for b in blocks if b.strip()]
        assert len(non_empty_blocks) >= 1, "Must have text content under headings"

        # No .txt file produced
        txt_path = result.md_path.with_suffix(".txt")
        assert not txt_path.exists(), ".txt file should not be generated"
