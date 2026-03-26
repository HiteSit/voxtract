"""Tests for the transcriber module."""

from pathlib import Path

import pytest

from mistral_voice_mcp.transcriber import MODEL_ID, transcribe
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
    async def test_saves_txt_output(self, wd_with_file, mock_mistral_client):
        wd, audio = wd_with_file
        result = await transcribe(mock_mistral_client, wd, audio)

        assert result.txt_path is not None
        assert result.txt_path.exists()
        assert result.txt_path.read_text() == "Transcribed text content."

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
    async def test_language_excludes_timestamps(
        self, wd_with_file, mock_mistral_client
    ):
        wd, audio = wd_with_file
        await transcribe(
            mock_mistral_client, wd, audio, language="en"
        )

        call_kwargs = mock_mistral_client.audio.transcriptions.complete_async.call_args
        assert call_kwargs.kwargs["language"] == "en"
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
