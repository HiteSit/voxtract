"""Tests for MCP server tools."""

import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mistral_voice_mcp.server import (
    _parse_bias_terms,
    set_workdir,
    get_workdir,
    list_inputs,
    list_transcriptions,
    read_transcription,
    set_context_bias,
    get_context_bias,
    clear_context_bias,
    transcribe_file,
)


@pytest.fixture
def mock_ctx(workdir: Path) -> MagicMock:
    """Mock Context with state storage."""
    state: dict = {}
    ctx = MagicMock(spec=["get_state", "set_state", "info", "report_progress"])
    ctx.get_state = AsyncMock(side_effect=lambda key: state.get(key))
    ctx.set_state = AsyncMock(
        side_effect=lambda key, value: state.__setitem__(key, value)
    )
    ctx.info = AsyncMock()
    ctx.report_progress = AsyncMock()
    return ctx


class TestParsebiasTerms:
    def test_comma_separated(self):
        assert _parse_bias_terms("ORCA, DFT, MCR") == ["ORCA", "DFT", "MCR"]

    def test_newline_separated(self):
        assert _parse_bias_terms("ORCA\nDFT\nMCR") == ["ORCA", "DFT", "MCR"]

    def test_mixed(self):
        result = _parse_bias_terms("ORCA, DFT\nMCR, Ugi")
        assert result == ["ORCA", "DFT", "MCR", "Ugi"]

    def test_empty_filtered(self):
        assert _parse_bias_terms("ORCA,,, DFT,  ,") == ["ORCA", "DFT"]


class TestWorkdirTools:
    @pytest.mark.asyncio
    async def test_set_workdir(self, tmp_path: Path, mock_ctx):
        result = await set_workdir(str(tmp_path), mock_ctx)
        assert "Work directory set to" in result
        assert (tmp_path / "input").is_dir()
        assert (tmp_path / "output").is_dir()
        mock_ctx.set_state.assert_called()

    @pytest.mark.asyncio
    async def test_get_workdir_without_set_errors(self, mock_ctx):
        with pytest.raises(ValueError, match="No work directory set"):
            await get_workdir(mock_ctx)

    @pytest.mark.asyncio
    async def test_get_workdir_after_set(self, tmp_path: Path, mock_ctx):
        await set_workdir(str(tmp_path), mock_ctx)
        result = await get_workdir(mock_ctx)
        assert "Work directory" in result
        assert "pending" in result


class TestInputOutputTools:
    @pytest.mark.asyncio
    async def test_list_inputs_empty(self, workdir: Path, mock_ctx):
        await set_workdir(str(workdir), mock_ctx)
        result = await list_inputs(mock_ctx)
        assert "No audio files" in result

    @pytest.mark.asyncio
    async def test_list_inputs_with_files(self, workdir: Path, mock_ctx):
        (workdir / "input" / "test.mp3").write_bytes(b"\x00")
        (workdir / "input" / "test2.wav").write_bytes(b"\x00")
        await set_workdir(str(workdir), mock_ctx)
        result = await list_inputs(mock_ctx)
        assert "2 audio files" in result
        assert "[pending]" in result

    @pytest.mark.asyncio
    async def test_list_transcriptions_empty(self, workdir: Path, mock_ctx):
        await set_workdir(str(workdir), mock_ctx)
        result = await list_transcriptions(mock_ctx)
        assert "No transcriptions" in result

    @pytest.mark.asyncio
    async def test_list_transcriptions_with_files(
        self, workdir: Path, mock_ctx
    ):
        (workdir / "output" / "test.json").write_text('{"text":"hello"}')
        (workdir / "output" / "test.md").write_text("## Speaker 1\n\nhello\n")
        await set_workdir(str(workdir), mock_ctx)
        result = await list_transcriptions(mock_ctx)
        assert "1 transcription" in result
        assert "[+md]" in result

    @pytest.mark.asyncio
    async def test_read_transcription(self, workdir: Path, mock_ctx):
        (workdir / "output" / "test.md").write_text("## Speaker 1\n\nHello world\n")
        await set_workdir(str(workdir), mock_ctx)
        result = await read_transcription("test.md", mock_ctx)
        assert "Hello world" in result

    @pytest.mark.asyncio
    async def test_read_transcription_not_found(
        self, workdir: Path, mock_ctx
    ):
        await set_workdir(str(workdir), mock_ctx)
        result = await read_transcription("nonexistent.md", mock_ctx)
        assert "not found" in result


class TestContextBiasTools:
    @pytest.mark.asyncio
    async def test_set_and_get(self, workdir: Path, mock_ctx):
        await set_workdir(str(workdir), mock_ctx)
        result = await set_context_bias("ORCA, DFT, MCR", mock_ctx)
        assert "3 terms" in result

        result = await get_context_bias(mock_ctx)
        assert "ORCA" in result
        assert "DFT" in result

    @pytest.mark.asyncio
    async def test_clear(self, workdir: Path, mock_ctx):
        await set_workdir(str(workdir), mock_ctx)
        await set_context_bias("ORCA", mock_ctx)
        result = await clear_context_bias(mock_ctx)
        assert "cleared" in result

        result = await get_context_bias(mock_ctx)
        assert "No context bias" in result

    @pytest.mark.asyncio
    async def test_set_from_file(self, workdir: Path, mock_ctx):
        await set_workdir(str(workdir), mock_ctx)
        terms_file = workdir / "my_terms.txt"
        terms_file.write_text("ORCA\nDFT\nMCR reaction\n")

        result = await set_context_bias("my_terms.txt", mock_ctx)
        assert "3 terms" in result


class TestTranscribeFile:
    @pytest.mark.asyncio
    async def test_file_not_found(self, workdir: Path, mock_ctx):
        await set_workdir(str(workdir), mock_ctx)
        result = await transcribe_file("nonexistent.mp3", mock_ctx)
        assert "not found" in result

    @pytest.mark.asyncio
    @patch("mistral_voice_mcp.server._get_client")
    async def test_transcribe_with_mock(
        self, mock_get_client, workdir: Path, mock_ctx, mock_mistral_client
    ):
        mock_get_client.return_value = mock_mistral_client

        (workdir / "input" / "test.mp3").write_bytes(b"\x00" * 100)
        await set_workdir(str(workdir), mock_ctx)

        result = await transcribe_file("test.mp3", mock_ctx)
        assert "Transcription complete" in result
        assert (workdir / "output" / "test.json").exists()
        assert (workdir / "output" / "test.md").exists()


@pytest.mark.integration
class TestIntegration:
    """Integration tests using real Mistral API. Run with: pytest -m integration"""

    @pytest.mark.asyncio
    async def test_transcribe_real_file(
        self, workdir_with_audio: Path, mock_ctx
    ):
        await set_workdir(str(workdir_with_audio), mock_ctx)
        result = await transcribe_file("Example.mpeg", mock_ctx)
        assert "Transcription complete" in result

        json_path = workdir_with_audio / "output" / "Example.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "text" in data
        assert len(data["text"]) > 0

    @pytest.mark.asyncio
    async def test_read_after_transcribe(
        self, workdir_with_audio: Path, mock_ctx
    ):
        await set_workdir(str(workdir_with_audio), mock_ctx)
        await transcribe_file("Example.mpeg", mock_ctx)

        result = await read_transcription("Example", mock_ctx, format="txt")
        assert len(result) > 50
