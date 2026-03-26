"""Tests for prompt templates."""

from mistral_voice_mcp.prompts import (
    clean_transcript_messages,
    meeting_notes_messages,
    summarize_transcript_messages,
    technical_notes_messages,
)

SAMPLE_TRANSCRIPT = "Speaker 1: The ORCA software does DFT calculations."


class TestPromptStructure:
    def test_clean_transcript_returns_messages(self):
        msgs = clean_transcript_messages(SAMPLE_TRANSCRIPT)
        assert isinstance(msgs, list)
        assert len(msgs) >= 1
        assert msgs[0]["role"] == "user"
        assert isinstance(msgs[0]["content"], str)

    def test_meeting_notes_returns_messages(self):
        msgs = meeting_notes_messages(SAMPLE_TRANSCRIPT)
        assert isinstance(msgs, list)
        assert msgs[0]["role"] == "user"

    def test_summarize_returns_messages(self):
        msgs = summarize_transcript_messages(SAMPLE_TRANSCRIPT)
        assert isinstance(msgs, list)
        assert msgs[0]["role"] == "user"

    def test_technical_notes_returns_messages(self):
        msgs = technical_notes_messages(SAMPLE_TRANSCRIPT)
        assert isinstance(msgs, list)
        assert msgs[0]["role"] == "user"

    def test_transcript_appears_in_content(self):
        for fn in [
            clean_transcript_messages,
            meeting_notes_messages,
            summarize_transcript_messages,
            technical_notes_messages,
        ]:
            msgs = fn(SAMPLE_TRANSCRIPT)
            assert SAMPLE_TRANSCRIPT in msgs[0]["content"]
