"""Tests for prompt templates."""

from mistral_voice_mcp.prompts import clean_transcript_messages

SAMPLE_TRANSCRIPT = (
    "## Speaker 1\n\n"
    "Um, so the ORCA software does like DFT calculations, you know?\n"
)


class TestCleanTranscriptPrompt:
    def test_returns_message_list(self):
        msgs = clean_transcript_messages(SAMPLE_TRANSCRIPT)
        assert isinstance(msgs, list)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert isinstance(msgs[0]["content"], str)

    def test_transcript_embedded_in_content(self):
        msgs = clean_transcript_messages(SAMPLE_TRANSCRIPT)
        assert SAMPLE_TRANSCRIPT in msgs[0]["content"]

    def test_contains_cleaning_instructions(self):
        msgs = clean_transcript_messages(SAMPLE_TRANSCRIPT)
        content = msgs[0]["content"]
        assert "filler words" in content
        assert "grammar" in content
        assert "logical flow" in content.lower() or "logic" in content.lower()

    def test_instructs_to_keep_speaker_headings(self):
        msgs = clean_transcript_messages(SAMPLE_TRANSCRIPT)
        content = msgs[0]["content"]
        assert "Speaker N" in content or "speaker" in content.lower()

    def test_instructs_not_to_add_information(self):
        msgs = clean_transcript_messages(SAMPLE_TRANSCRIPT)
        content = msgs[0]["content"]
        assert "not add information" in content.lower() or "do not add" in content.lower()
