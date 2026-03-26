"""Tests for MCP server tools."""

import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mistral_voice_mcp.server import (
    SUPPORTED_LANGUAGES,
    _parse_bias_terms,
    set_workdir,
    get_workdir,
    set_language,
    get_language,
    list_inbox,
    create_session_tool,
    list_sessions_tool,
    transcribe_session,
    read_staging_transcript,
    finalize_session,
    list_recordings,
    read_transcript,
    save_processed,
    set_context_bias,
    get_context_bias,
    clear_context_bias,
)


@pytest.fixture
def mock_ctx():
    """Mock MCP Context with state storage."""
    ctx = MagicMock()
    state: dict[str, str] = {}
    ctx.get_state = AsyncMock(side_effect=lambda k: state.get(k))
    ctx.set_state = AsyncMock(side_effect=lambda k, v: state.__setitem__(k, v))
    ctx.info = AsyncMock()
    ctx.report_progress = AsyncMock()
    return ctx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestParseBiasTerms:
    def test_comma_separated(self):
        assert _parse_bias_terms("A, B, C") == ["A", "B", "C"]

    def test_newline_separated(self):
        assert _parse_bias_terms("A\nB\nC") == ["A", "B", "C"]

    def test_mixed(self):
        assert _parse_bias_terms("A, B\nC") == ["A", "B", "C"]

    def test_empty_filtered(self):
        assert _parse_bias_terms(", , A, ") == ["A"]


# ---------------------------------------------------------------------------
# Workdir Tools
# ---------------------------------------------------------------------------


class TestWorkdirTools:
    @pytest.mark.asyncio
    async def test_set_workdir(self, workdir: Path, mock_ctx):
        result = await set_workdir(str(workdir), mock_ctx)
        assert "Work directory set to:" in result
        assert "inbox/" in result

    @pytest.mark.asyncio
    async def test_get_workdir_without_set_errors(self, mock_ctx):
        with pytest.raises(ValueError, match="No work directory set"):
            await get_workdir(mock_ctx)

    @pytest.mark.asyncio
    async def test_get_workdir_after_set(self, workdir: Path, mock_ctx):
        await set_workdir(str(workdir), mock_ctx)
        result = await get_workdir(mock_ctx)
        assert "Work directory:" in result
        assert "inbox/" in result
        assert "recordings" in result


# ---------------------------------------------------------------------------
# Context Bias Tools
# ---------------------------------------------------------------------------


class TestContextBiasTools:
    @pytest.mark.asyncio
    async def test_set_and_get(self, workdir: Path, mock_ctx):
        await set_workdir(str(workdir), mock_ctx)
        await set_context_bias("ORCA, DFT, MCR", mock_ctx)
        result = await get_context_bias(mock_ctx)
        assert "3 terms" in result
        assert "ORCA" in result

    @pytest.mark.asyncio
    async def test_clear(self, workdir: Path, mock_ctx):
        await set_workdir(str(workdir), mock_ctx)
        await set_context_bias("ORCA", mock_ctx)
        result = await clear_context_bias(mock_ctx)
        assert "cleared" in result.lower()

    @pytest.mark.asyncio
    async def test_set_from_file(self, workdir: Path, mock_ctx):
        await set_workdir(str(workdir), mock_ctx)
        terms_file = workdir / "my_terms.txt"
        terms_file.write_text("ORCA\nDFT\nMCR reaction\n")
        result = await set_context_bias("my_terms.txt", mock_ctx)
        assert "3 terms" in result


# ---------------------------------------------------------------------------
# Language Tools
# ---------------------------------------------------------------------------


class TestLanguageTools:
    @pytest.mark.asyncio
    async def test_default_is_english(self, mock_ctx):
        result = await get_language(mock_ctx)
        assert "english" in result.lower()
        assert "en" in result
        assert "enabled" in result.lower()

    @pytest.mark.asyncio
    async def test_set_by_name(self, mock_ctx):
        result = await set_language("italian", mock_ctx)
        assert "it" in result
        assert "disabled" in result.lower()

    @pytest.mark.asyncio
    async def test_set_by_code(self, mock_ctx):
        result = await set_language("fr", mock_ctx)
        assert "fr" in result
        assert "disabled" in result.lower()

    @pytest.mark.asyncio
    async def test_set_english_enables_timestamps(self, mock_ctx):
        await set_language("italian", mock_ctx)
        result = await set_language("english", mock_ctx)
        assert "enabled" in result.lower()

    @pytest.mark.asyncio
    async def test_case_insensitive(self, mock_ctx):
        result = await set_language("GERMAN", mock_ctx)
        assert "de" in result

    @pytest.mark.asyncio
    async def test_unsupported_language_rejected(self, mock_ctx):
        result = await set_language("klingon", mock_ctx)
        assert "Unsupported" in result

    @pytest.mark.asyncio
    async def test_get_after_set(self, mock_ctx):
        await set_language("japanese", mock_ctx)
        result = await get_language(mock_ctx)
        assert "japanese" in result.lower()
        assert "ja" in result
        assert "disabled" in result.lower()

    def test_all_codes_map_to_valid_entries(self):
        codes = set(SUPPORTED_LANGUAGES.values())
        assert len(codes) == 13


# ---------------------------------------------------------------------------
# Inbox Tools
# ---------------------------------------------------------------------------


class TestListInbox:
    @pytest.mark.asyncio
    async def test_empty(self, workdir: Path, mock_ctx):
        await set_workdir(str(workdir), mock_ctx)
        result = await list_inbox(mock_ctx)
        assert "No audio files" in result

    @pytest.mark.asyncio
    async def test_with_files(self, workdir: Path, mock_ctx):
        (workdir / "inbox" / "test.mp3").write_bytes(b"\x00" * 100)
        await set_workdir(str(workdir), mock_ctx)
        result = await list_inbox(mock_ctx)
        assert "1 audio files" in result
        assert "test.mp3" in result


# ---------------------------------------------------------------------------
# Session Tools
# ---------------------------------------------------------------------------


class TestCreateSession:
    @pytest.mark.asyncio
    async def test_create_with_all_files(self, workdir: Path, mock_ctx):
        (workdir / "inbox" / "a.mpeg").write_bytes(b"\x00" * 100)
        (workdir / "inbox" / "b.mpeg").write_bytes(b"\x00" * 100)
        await set_workdir(str(workdir), mock_ctx)
        result = await create_session_tool(mock_ctx)
        assert "Session created:" in result
        assert "Files: 2" in result

    @pytest.mark.asyncio
    async def test_create_with_specific_files(self, workdir: Path, mock_ctx):
        (workdir / "inbox" / "a.mpeg").write_bytes(b"\x00" * 100)
        (workdir / "inbox" / "b.mpeg").write_bytes(b"\x00" * 100)
        await set_workdir(str(workdir), mock_ctx)
        result = await create_session_tool(mock_ctx, filenames=["a.mpeg"])
        assert "Files: 1" in result
        assert "a.mpeg" in result

    @pytest.mark.asyncio
    async def test_create_missing_file(self, workdir: Path, mock_ctx):
        await set_workdir(str(workdir), mock_ctx)
        result = await create_session_tool(mock_ctx, filenames=["nope.mp3"])
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_create_empty_inbox(self, workdir: Path, mock_ctx):
        await set_workdir(str(workdir), mock_ctx)
        result = await create_session_tool(mock_ctx)
        assert "No audio files" in result


class TestListSessions:
    @pytest.mark.asyncio
    async def test_no_sessions(self, workdir: Path, mock_ctx):
        await set_workdir(str(workdir), mock_ctx)
        result = await list_sessions_tool(mock_ctx)
        assert "No active" in result

    @pytest.mark.asyncio
    async def test_with_sessions(self, workdir: Path, mock_ctx):
        (workdir / "inbox" / "test.mpeg").write_bytes(b"\x00" * 100)
        await set_workdir(str(workdir), mock_ctx)
        await create_session_tool(mock_ctx)
        result = await list_sessions_tool(mock_ctx)
        assert "1 staging sessions" in result
        assert "pending" in result


# ---------------------------------------------------------------------------
# Transcribe Session
# ---------------------------------------------------------------------------


class TestTranscribeSession:
    @pytest.mark.asyncio
    @patch("mistral_voice_mcp.server._get_client")
    async def test_transcribes_all_files(
        self, mock_get_client, workdir: Path, mock_ctx, mock_mistral_client
    ):
        mock_get_client.return_value = mock_mistral_client
        (workdir / "inbox" / "test.mpeg").write_bytes(b"\x00" * 100)
        await set_workdir(str(workdir), mock_ctx)

        # Create session
        create_result = await create_session_tool(mock_ctx)
        session_id = create_result.split("\n")[0].split(": ")[1]

        result = await transcribe_session(session_id, mock_ctx)
        assert "Transcription complete: 1/1" in result
        assert "test.mpeg" in result

    @pytest.mark.asyncio
    @patch("mistral_voice_mcp.server._get_client")
    async def test_uses_session_language(
        self, mock_get_client, workdir: Path, mock_ctx, mock_mistral_client
    ):
        mock_get_client.return_value = mock_mistral_client
        (workdir / "inbox" / "test.mpeg").write_bytes(b"\x00" * 100)
        await set_workdir(str(workdir), mock_ctx)
        await set_language("italian", mock_ctx)

        create_result = await create_session_tool(mock_ctx)
        session_id = create_result.split("\n")[0].split(": ")[1]

        await transcribe_session(session_id, mock_ctx)
        call_kwargs = mock_mistral_client.audio.transcriptions.complete_async.call_args
        assert call_kwargs.kwargs["language"] == "it"


# ---------------------------------------------------------------------------
# Read Staging Transcript
# ---------------------------------------------------------------------------


class TestReadStagingTranscript:
    @pytest.mark.asyncio
    @patch("mistral_voice_mcp.server._get_client")
    async def test_read_after_transcribe(
        self, mock_get_client, workdir: Path, mock_ctx, mock_mistral_client
    ):
        mock_get_client.return_value = mock_mistral_client
        (workdir / "inbox" / "test.mpeg").write_bytes(b"\x00" * 100)
        await set_workdir(str(workdir), mock_ctx)

        create_result = await create_session_tool(mock_ctx)
        session_id = create_result.split("\n")[0].split(": ")[1]

        await transcribe_session(session_id, mock_ctx)
        result = await read_staging_transcript(session_id, mock_ctx)
        assert "Speaker 1" in result

    @pytest.mark.asyncio
    async def test_not_transcribed_yet(self, workdir: Path, mock_ctx):
        (workdir / "inbox" / "test.mpeg").write_bytes(b"\x00" * 100)
        await set_workdir(str(workdir), mock_ctx)

        create_result = await create_session_tool(mock_ctx)
        session_id = create_result.split("\n")[0].split(": ")[1]

        result = await read_staging_transcript(session_id, mock_ctx)
        assert "not fully transcribed" in result


# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------


class TestFinalize:
    @pytest.mark.asyncio
    @patch("mistral_voice_mcp.server._get_client")
    async def test_full_finalize(
        self, mock_get_client, workdir: Path, mock_ctx, mock_mistral_client
    ):
        mock_get_client.return_value = mock_mistral_client
        (workdir / "inbox" / "test.mpeg").write_bytes(b"\x00" * 100)
        await set_workdir(str(workdir), mock_ctx)

        create_result = await create_session_tool(mock_ctx)
        session_id = create_result.split("\n")[0].split(": ")[1]

        await transcribe_session(session_id, mock_ctx)
        result = await finalize_session(session_id, "Test Recording", mock_ctx)

        assert "Finalized: test-recording/" in result
        assert (workdir / "test-recording" / "test.mpeg").exists()
        assert (workdir / "test-recording" / "transcript.md").exists()
        # Staging cleaned up
        assert not (workdir / ".staging" / session_id).exists()

    @pytest.mark.asyncio
    @patch("mistral_voice_mcp.server._get_client")
    async def test_cleans_inbox(
        self, mock_get_client, workdir: Path, mock_ctx, mock_mistral_client
    ):
        mock_get_client.return_value = mock_mistral_client
        (workdir / "inbox" / "test.mpeg").write_bytes(b"\x00" * 100)
        await set_workdir(str(workdir), mock_ctx)

        create_result = await create_session_tool(mock_ctx)
        session_id = create_result.split("\n")[0].split(": ")[1]

        await transcribe_session(session_id, mock_ctx)
        await finalize_session(session_id, "My Recording", mock_ctx)

        assert not (workdir / "inbox" / "test.mpeg").exists()

    @pytest.mark.asyncio
    async def test_not_transcribed_rejects(self, workdir: Path, mock_ctx):
        (workdir / "inbox" / "test.mpeg").write_bytes(b"\x00" * 100)
        await set_workdir(str(workdir), mock_ctx)

        create_result = await create_session_tool(mock_ctx)
        session_id = create_result.split("\n")[0].split(": ")[1]

        result = await finalize_session(session_id, "Nope", mock_ctx)
        assert "not fully transcribed" in result


# ---------------------------------------------------------------------------
# Recordings
# ---------------------------------------------------------------------------


class TestListRecordings:
    @pytest.mark.asyncio
    async def test_no_recordings(self, workdir: Path, mock_ctx):
        await set_workdir(str(workdir), mock_ctx)
        result = await list_recordings(mock_ctx)
        assert "No finalized recordings" in result

    @pytest.mark.asyncio
    async def test_with_recordings(self, workdir: Path, mock_ctx):
        rec = workdir / "my-recording"
        rec.mkdir()
        (rec / "audio.mpeg").write_bytes(b"\x00")
        (rec / "transcript.md").write_text("## Speaker 1\n\nHello.\n")
        await set_workdir(str(workdir), mock_ctx)
        result = await list_recordings(mock_ctx)
        assert "1 recordings" in result
        assert "my-recording/" in result
        assert "transcript" in result


class TestReadTranscript:
    @pytest.mark.asyncio
    async def test_read_transcript(self, workdir: Path, mock_ctx):
        rec = workdir / "my-recording"
        rec.mkdir()
        (rec / "transcript.md").write_text("## Speaker 1\n\nHello.\n")
        await set_workdir(str(workdir), mock_ctx)
        result = await read_transcript("my-recording", mock_ctx)
        assert "Hello." in result

    @pytest.mark.asyncio
    async def test_read_clean_transcript(self, workdir: Path, mock_ctx):
        rec = workdir / "my-recording"
        rec.mkdir()
        (rec / "transcript_clean.md").write_text("Cleaned version.")
        await set_workdir(str(workdir), mock_ctx)
        result = await read_transcript("my-recording", mock_ctx, clean=True)
        assert "Cleaned version." in result

    @pytest.mark.asyncio
    async def test_recording_not_found(self, workdir: Path, mock_ctx):
        await set_workdir(str(workdir), mock_ctx)
        result = await read_transcript("nonexistent", mock_ctx)
        assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# Save Processed
# ---------------------------------------------------------------------------


class TestSaveProcessed:
    @pytest.mark.asyncio
    async def test_saves_clean_file(self, workdir: Path, mock_ctx):
        rec = workdir / "my-recording"
        rec.mkdir()
        (rec / "transcript.md").write_text("raw")
        await set_workdir(str(workdir), mock_ctx)
        result = await save_processed("my-recording", "cleaned", mock_ctx)
        assert "Saved" in result
        assert (rec / "transcript_clean.md").read_text() == "cleaned"

    @pytest.mark.asyncio
    async def test_recording_not_found(self, workdir: Path, mock_ctx):
        await set_workdir(str(workdir), mock_ctx)
        result = await save_processed("nonexistent", "content", mock_ctx)
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_no_workdir_raises(self, mock_ctx):
        with pytest.raises(ValueError, match="No work directory set"):
            await save_processed("rec", "content", mock_ctx)
