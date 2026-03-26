"""Tests for the staging module."""

import shutil
from pathlib import Path

import pytest

from mistral_voice_mcp.staging import (
    StagingSession,
    create_session,
    get_session,
    list_sessions,
    slugify,
)


# ---------------------------------------------------------------------------
# Slugify
# ---------------------------------------------------------------------------


class TestSlugify:
    def test_lowercase(self):
        assert slugify("Hello World") == "hello-world"

    def test_spaces_to_hyphens(self):
        assert slugify("one two three") == "one-two-three"

    def test_underscores_to_hyphens(self):
        assert slugify("one_two_three") == "one-two-three"

    def test_special_chars_removed(self):
        assert slugify("ORCA Software: DFT!") == "orca-software-dft"

    def test_multiple_hyphens_collapsed(self):
        assert slugify("a - - b") == "a-b"

    def test_leading_trailing_hyphens_stripped(self):
        assert slugify("--hello--") == "hello"

    def test_max_length_truncation(self):
        result = slugify("a" * 100, max_length=80)
        assert len(result) <= 80

    def test_max_length_no_trailing_hyphen(self):
        # "a-b-c-d-..." truncated should not end with hyphen
        long_name = "-".join(["word"] * 30)
        result = slugify(long_name, max_length=20)
        assert not result.endswith("-")
        assert len(result) <= 20

    def test_empty_string_returns_empty(self):
        assert slugify("") == ""

    def test_only_special_chars_returns_empty(self):
        assert slugify("!!!???") == ""

    def test_unicode_stripped(self):
        assert slugify("café résumé") == "caf-rsum"

    def test_realistic_transcript_name(self):
        assert (
            slugify("ChromSword vs DryLab Method Optimization")
            == "chromsword-vs-drylab-method-optimization"
        )


# ---------------------------------------------------------------------------
# StagingSession
# ---------------------------------------------------------------------------


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """Create a workdir with inbox/ and .staging/."""
    (tmp_path / "inbox").mkdir()
    (tmp_path / ".staging").mkdir()
    return tmp_path


@pytest.fixture
def audio_files(workdir: Path) -> list[Path]:
    """Create two dummy audio files in inbox/."""
    files = []
    for name in ["part1.mpeg", "part2.mpeg"]:
        f = workdir / "inbox" / name
        f.write_bytes(b"\x00" * 100)
        files.append(f)
    return files


class TestStagingSession:
    def test_creates_session_dir(self, workdir: Path):
        session = StagingSession(workdir)
        assert session.session_dir.exists()
        assert session.session_dir.parent.name == ".staging"

    def test_custom_session_id(self, workdir: Path):
        session = StagingSession(workdir, session_id="test123")
        assert session.session_id == "test123"
        assert session.session_dir.name == "test123"

    def test_audio_files_empty(self, workdir: Path):
        session = StagingSession(workdir)
        assert session.audio_files == []

    def test_audio_files_sorted(self, workdir: Path):
        session = StagingSession(workdir)
        (session.session_dir / "b.mpeg").write_bytes(b"\x00")
        (session.session_dir / "a.mpeg").write_bytes(b"\x00")
        names = [p.name for p in session.audio_files]
        assert names == ["a.mpeg", "b.mpeg"]

    def test_transcript_files(self, workdir: Path):
        session = StagingSession(workdir)
        (session.session_dir / "part1.md").write_text("## Speaker 1\n\nHello\n")
        assert len(session.transcript_files) == 1

    def test_is_fully_transcribed_false(self, workdir: Path):
        session = StagingSession(workdir)
        (session.session_dir / "part1.mpeg").write_bytes(b"\x00")
        assert not session.is_fully_transcribed()

    def test_is_fully_transcribed_true(self, workdir: Path):
        session = StagingSession(workdir)
        (session.session_dir / "part1.mpeg").write_bytes(b"\x00")
        (session.session_dir / "part1.md").write_text("transcript")
        assert session.is_fully_transcribed()

    def test_is_fully_transcribed_no_audio(self, workdir: Path):
        session = StagingSession(workdir)
        assert not session.is_fully_transcribed()


class TestMergeTranscripts:
    def test_empty_no_transcripts(self, workdir: Path):
        session = StagingSession(workdir)
        assert session.merge_transcripts() == ""

    def test_single_file_no_part_header(self, workdir: Path):
        session = StagingSession(workdir)
        content = "## Speaker 1\n\nHello there.\n"
        (session.session_dir / "example.mpeg").write_bytes(b"\x00")
        (session.session_dir / "example.md").write_text(content)
        assert session.merge_transcripts() == content

    def test_multiple_files_adds_part_headers(self, workdir: Path):
        session = StagingSession(workdir)
        (session.session_dir / "part1.mpeg").write_bytes(b"\x00")
        (session.session_dir / "part1.md").write_text("## Speaker 1\n\nFirst part.\n")
        (session.session_dir / "part2.mpeg").write_bytes(b"\x00")
        (session.session_dir / "part2.md").write_text("## Speaker 1\n\nSecond part.\n")

        merged = session.merge_transcripts()
        assert "# Part: part1.mpeg" in merged
        assert "# Part: part2.mpeg" in merged
        assert "First part." in merged
        assert "Second part." in merged

    def test_files_ordered_alphabetically(self, workdir: Path):
        session = StagingSession(workdir)
        (session.session_dir / "b_rec.mpeg").write_bytes(b"\x00")
        (session.session_dir / "b_rec.md").write_text("## Speaker 1\n\nB content.\n")
        (session.session_dir / "a_rec.mpeg").write_bytes(b"\x00")
        (session.session_dir / "a_rec.md").write_text("## Speaker 1\n\nA content.\n")

        merged = session.merge_transcripts()
        pos_a = merged.index("# Part: a_rec.mpeg")
        pos_b = merged.index("# Part: b_rec.mpeg")
        assert pos_a < pos_b

    def test_preserves_speaker_headings(self, workdir: Path):
        session = StagingSession(workdir)
        (session.session_dir / "rec.mpeg").write_bytes(b"\x00")
        (session.session_dir / "rec.md").write_text(
            "## Speaker 1\n\nHello.\n\n## Speaker 2\n\nHi.\n"
        )
        merged = session.merge_transcripts()
        assert "## Speaker 1" in merged
        assert "## Speaker 2" in merged


class TestFinalize:
    def test_creates_named_directory(self, workdir: Path):
        session = StagingSession(workdir)
        (session.session_dir / "test.mpeg").write_bytes(b"\x00")
        (session.session_dir / "test.md").write_text("## Speaker 1\n\nContent.\n")

        final = session.finalize("Test Recording")
        assert final.name == "test-recording"
        assert final.exists()

    def test_moves_audio_files(self, workdir: Path):
        session = StagingSession(workdir)
        (session.session_dir / "test.mpeg").write_bytes(b"\x00" * 50)
        (session.session_dir / "test.md").write_text("## Speaker 1\n\nContent.\n")

        final = session.finalize("My Recording")
        assert (final / "test.mpeg").exists()
        assert (final / "test.mpeg").stat().st_size == 50

    def test_writes_merged_transcript(self, workdir: Path):
        session = StagingSession(workdir)
        (session.session_dir / "test.mpeg").write_bytes(b"\x00")
        (session.session_dir / "test.md").write_text("## Speaker 1\n\nContent.\n")

        final = session.finalize("My Recording")
        transcript = final / "transcript.md"
        assert transcript.exists()
        assert "Content." in transcript.read_text()

    def test_removes_staging_directory(self, workdir: Path):
        session = StagingSession(workdir)
        session_dir = session.session_dir
        (session_dir / "test.mpeg").write_bytes(b"\x00")
        (session_dir / "test.md").write_text("content")

        session.finalize("My Recording")
        assert not session_dir.exists()

    def test_slugifies_name(self, workdir: Path):
        session = StagingSession(workdir)
        (session.session_dir / "test.mpeg").write_bytes(b"\x00")
        (session.session_dir / "test.md").write_text("content")

        final = session.finalize("ORCA Software: DFT Calculations!")
        assert final.name == "orca-software-dft-calculations"

    def test_rejects_empty_name(self, workdir: Path):
        session = StagingSession(workdir)
        with pytest.raises(ValueError, match="empty"):
            session.finalize("")

    def test_rejects_duplicate_name(self, workdir: Path):
        # Create existing directory
        (workdir / "existing-name").mkdir()

        session = StagingSession(workdir)
        (session.session_dir / "test.mpeg").write_bytes(b"\x00")
        (session.session_dir / "test.md").write_text("content")

        with pytest.raises(ValueError, match="already exists"):
            session.finalize("Existing Name")

    def test_multi_file_finalize(self, workdir: Path):
        session = StagingSession(workdir)
        (session.session_dir / "part1.mpeg").write_bytes(b"\x00")
        (session.session_dir / "part1.md").write_text("## Speaker 1\n\nFirst.\n")
        (session.session_dir / "part2.mpeg").write_bytes(b"\x00")
        (session.session_dir / "part2.md").write_text("## Speaker 1\n\nSecond.\n")

        final = session.finalize("Multi Part Recording")
        assert (final / "part1.mpeg").exists()
        assert (final / "part2.mpeg").exists()
        transcript = (final / "transcript.md").read_text()
        assert "# Part: part1.mpeg" in transcript
        assert "# Part: part2.mpeg" in transcript


# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------


class TestCreateSession:
    def test_copies_audio_files(self, workdir: Path, audio_files: list[Path]):
        session = create_session(workdir, audio_files)
        assert len(session.audio_files) == 2
        # Originals still exist
        for f in audio_files:
            assert f.exists()

    def test_single_file(self, workdir: Path, audio_files: list[Path]):
        session = create_session(workdir, audio_files[:1])
        assert len(session.audio_files) == 1

    def test_empty_list_raises(self, workdir: Path):
        with pytest.raises(ValueError, match="At least one"):
            create_session(workdir, [])

    def test_missing_file_raises(self, workdir: Path):
        with pytest.raises(FileNotFoundError):
            create_session(workdir, [Path("/nonexistent/audio.mp3")])


class TestListSessions:
    def test_empty(self, workdir: Path):
        assert list_sessions(workdir) == []

    def test_lists_sessions(self, workdir: Path, audio_files: list[Path]):
        create_session(workdir, audio_files[:1])
        create_session(workdir, audio_files[1:])
        sessions = list_sessions(workdir)
        assert len(sessions) == 2

    def test_sorted_by_id(self, workdir: Path):
        StagingSession(workdir, "bbb")
        StagingSession(workdir, "aaa")
        sessions = list_sessions(workdir)
        assert sessions[0].session_id == "aaa"
        assert sessions[1].session_id == "bbb"


class TestGetSession:
    def test_existing_session(self, workdir: Path, audio_files: list[Path]):
        created = create_session(workdir, audio_files)
        retrieved = get_session(workdir, created.session_id)
        assert retrieved.session_id == created.session_id

    def test_nonexistent_raises(self, workdir: Path):
        with pytest.raises(ValueError, match="not found"):
            get_session(workdir, "nonexistent")
