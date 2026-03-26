"""End-to-end workflow tests exercising the full staging → finalize pipeline."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mistral_voice_mcp.server import (
    create_session_tool,
    finalize_session,
    list_inbox,
    list_recordings,
    read_staging_transcript,
    read_transcript,
    save_processed,
    set_workdir,
    transcribe_session,
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


class TestSingleFileWorkflow:
    """Full workflow: inbox → create session → transcribe → finalize."""

    @pytest.mark.asyncio
    @patch("mistral_voice_mcp.server._get_client")
    async def test_single_file_e2e(
        self, mock_get_client, workdir: Path, mock_ctx, mock_mistral_client
    ):
        mock_get_client.return_value = mock_mistral_client

        # 1. Place audio in inbox
        (workdir / "inbox" / "example.mpeg").write_bytes(b"\x00" * 100)
        await set_workdir(str(workdir), mock_ctx)

        # 2. Verify inbox
        inbox_result = await list_inbox(mock_ctx)
        assert "example.mpeg" in inbox_result

        # 3. Create session
        create_result = await create_session_tool(mock_ctx)
        session_id = create_result.split("\n")[0].split(": ")[1]
        assert session_id

        # 4. Transcribe
        transcribe_result = await transcribe_session(session_id, mock_ctx)
        assert "1/1" in transcribe_result

        # 5. Read staging transcript
        staging_text = await read_staging_transcript(session_id, mock_ctx)
        assert "Speaker 1" in staging_text

        # 6. Finalize
        finalize_result = await finalize_session(
            session_id, "Test Single Recording", mock_ctx
        )
        assert "Finalized: test-single-recording/" in finalize_result

        # 7. Verify final structure
        final_dir = workdir / "test-single-recording"
        assert final_dir.is_dir()
        assert (final_dir / "example.mpeg").exists()
        assert (final_dir / "transcript.md").exists()
        assert "Speaker 1" in (final_dir / "transcript.md").read_text()

        # 8. Inbox cleaned
        assert not (workdir / "inbox" / "example.mpeg").exists()

        # 9. Staging cleaned
        assert not (workdir / ".staging" / session_id).exists()

        # 10. Shows up in recordings list
        recordings_result = await list_recordings(mock_ctx)
        assert "test-single-recording/" in recordings_result

        # 11. Can read transcript via tool
        transcript = await read_transcript("test-single-recording", mock_ctx)
        assert "Speaker 1" in transcript


class TestMultiFileWorkflow:
    """Full workflow with multiple audio files → single merged transcript."""

    @pytest.mark.asyncio
    @patch("mistral_voice_mcp.server._get_client")
    async def test_multi_file_e2e(
        self, mock_get_client, workdir: Path, mock_ctx, mock_mistral_client
    ):
        mock_get_client.return_value = mock_mistral_client

        # 1. Place two audio files in inbox
        (workdir / "inbox" / "part1.mpeg").write_bytes(b"\x00" * 100)
        (workdir / "inbox" / "part2.mpeg").write_bytes(b"\x00" * 100)
        await set_workdir(str(workdir), mock_ctx)

        # 2. Create session with both
        create_result = await create_session_tool(mock_ctx)
        session_id = create_result.split("\n")[0].split(": ")[1]
        assert "Files: 2" in create_result

        # 3. Transcribe
        transcribe_result = await transcribe_session(session_id, mock_ctx)
        assert "2/2" in transcribe_result

        # 4. Read merged staging transcript — should have Part headers
        staging_text = await read_staging_transcript(session_id, mock_ctx)
        assert "# Part: part1.mpeg" in staging_text
        assert "# Part: part2.mpeg" in staging_text

        # 5. Finalize
        finalize_result = await finalize_session(
            session_id, "Multi Part Recording", mock_ctx
        )
        assert "Finalized: multi-part-recording/" in finalize_result

        # 6. Verify final structure
        final_dir = workdir / "multi-part-recording"
        assert (final_dir / "part1.mpeg").exists()
        assert (final_dir / "part2.mpeg").exists()
        transcript = (final_dir / "transcript.md").read_text()
        assert "# Part: part1.mpeg" in transcript
        assert "# Part: part2.mpeg" in transcript

        # 7. Both inbox files cleaned
        assert not (workdir / "inbox" / "part1.mpeg").exists()
        assert not (workdir / "inbox" / "part2.mpeg").exists()


class TestCleanAfterFinalize:
    """Full workflow + post-processing with save_processed."""

    @pytest.mark.asyncio
    @patch("mistral_voice_mcp.server._get_client")
    async def test_clean_transcript_after_finalize(
        self, mock_get_client, workdir: Path, mock_ctx, mock_mistral_client
    ):
        mock_get_client.return_value = mock_mistral_client

        # Full workflow
        (workdir / "inbox" / "example.mpeg").write_bytes(b"\x00" * 100)
        await set_workdir(str(workdir), mock_ctx)

        create_result = await create_session_tool(mock_ctx)
        session_id = create_result.split("\n")[0].split(": ")[1]
        await transcribe_session(session_id, mock_ctx)
        await finalize_session(session_id, "To Be Cleaned", mock_ctx)

        # Save cleaned version
        cleaned_content = "## Speaker 1\n\nCleaned text here.\n"
        result = await save_processed("to-be-cleaned", cleaned_content, mock_ctx)
        assert "Saved" in result

        # Verify both files exist
        rec_dir = workdir / "to-be-cleaned"
        assert (rec_dir / "transcript.md").exists()
        assert (rec_dir / "transcript_clean.md").exists()
        assert (rec_dir / "transcript_clean.md").read_text() == cleaned_content

        # Can read both via tool
        raw = await read_transcript("to-be-cleaned", mock_ctx)
        assert "Speaker 1" in raw
        clean = await read_transcript("to-be-cleaned", mock_ctx, clean=True)
        assert "Cleaned text here." in clean
