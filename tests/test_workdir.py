"""Tests for WorkDirectory class."""

from pathlib import Path

import pytest

from mistral_voice_mcp.workdir import MAX_CONTEXT_BIAS_TERMS, WorkDirectory


class TestInit:
    def test_creates_subdirs(self, tmp_path: Path):
        wd = WorkDirectory(tmp_path / "new_project")
        assert wd.inbox_dir.is_dir()
        assert wd.staging_dir.is_dir()

    def test_existing_dirs_no_error(self, tmp_path: Path):
        root = tmp_path / "project"
        root.mkdir()
        (root / "inbox").mkdir()
        (root / ".staging").mkdir()
        wd = WorkDirectory(root)
        assert wd.root == root


class TestScanInbox:
    def test_empty(self, tmp_path: Path):
        wd = WorkDirectory(tmp_path)
        assert wd.scan_inbox() == []

    def test_filters_by_extension(self, tmp_path: Path):
        wd = WorkDirectory(tmp_path)
        (wd.inbox_dir / "audio.mp3").write_bytes(b"\x00")
        (wd.inbox_dir / "notes.txt").write_text("hello")
        (wd.inbox_dir / "data.csv").write_text("a,b")
        results = wd.scan_inbox()
        assert len(results) == 1
        assert results[0].name == "audio.mp3"

    def test_recursive(self, tmp_path: Path):
        wd = WorkDirectory(tmp_path)
        sub = wd.inbox_dir / "session1"
        sub.mkdir()
        (sub / "recording.wav").write_bytes(b"\x00")
        (wd.inbox_dir / "main.flac").write_bytes(b"\x00")
        results = wd.scan_inbox()
        assert len(results) == 2

    def test_all_extensions(self, tmp_path: Path):
        wd = WorkDirectory(tmp_path)
        for ext in [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mpeg"]:
            (wd.inbox_dir / f"file{ext}").write_bytes(b"\x00")
        assert len(wd.scan_inbox()) == 6


class TestScanRecordings:
    def test_empty(self, tmp_path: Path):
        wd = WorkDirectory(tmp_path)
        assert wd.scan_recordings() == []

    def test_finds_recording_dirs(self, tmp_path: Path):
        wd = WorkDirectory(tmp_path)
        (tmp_path / "my-recording").mkdir()
        (tmp_path / "another-one").mkdir()
        results = wd.scan_recordings()
        assert len(results) == 2
        assert results[0].name == "another-one"
        assert results[1].name == "my-recording"

    def test_excludes_inbox_and_staging(self, tmp_path: Path):
        wd = WorkDirectory(tmp_path)
        (tmp_path / "my-recording").mkdir()
        results = wd.scan_recordings()
        assert len(results) == 1
        assert results[0].name == "my-recording"

    def test_excludes_hidden_dirs(self, tmp_path: Path):
        wd = WorkDirectory(tmp_path)
        (tmp_path / ".hidden").mkdir()
        (tmp_path / "visible").mkdir()
        results = wd.scan_recordings()
        assert len(results) == 1
        assert results[0].name == "visible"

    def test_ignores_files(self, tmp_path: Path):
        wd = WorkDirectory(tmp_path)
        (tmp_path / "not-a-dir.txt").write_text("hello")
        (tmp_path / "my-recording").mkdir()
        results = wd.scan_recordings()
        assert len(results) == 1


class TestStatus:
    def test_all_zeros(self, tmp_path: Path):
        wd = WorkDirectory(tmp_path)
        assert wd.status() == {"inbox": 0, "staging": 0, "recordings": 0}

    def test_counts(self, tmp_path: Path):
        wd = WorkDirectory(tmp_path)
        (wd.inbox_dir / "audio.mp3").write_bytes(b"\x00")
        (wd.inbox_dir / "audio2.wav").write_bytes(b"\x00")
        (wd.staging_dir / "session1").mkdir()
        (tmp_path / "finished-recording").mkdir()
        assert wd.status() == {"inbox": 2, "staging": 1, "recordings": 1}


class TestContextBias:
    def test_save_and_load_roundtrip(self, tmp_path: Path):
        wd = WorkDirectory(tmp_path)
        terms = ["ORCA", "DFT", "MCR reaction"]
        wd.save_context_bias(terms)
        loaded = wd.load_context_bias()
        assert loaded == terms

    def test_max_terms_exceeded(self, tmp_path: Path):
        wd = WorkDirectory(tmp_path)
        terms = [f"term_{i}" for i in range(MAX_CONTEXT_BIAS_TERMS + 1)]
        with pytest.raises(ValueError, match="at most"):
            wd.save_context_bias(terms)

    def test_max_terms_exact_limit(self, tmp_path: Path):
        wd = WorkDirectory(tmp_path)
        terms = [f"term_{i}" for i in range(MAX_CONTEXT_BIAS_TERMS)]
        wd.save_context_bias(terms)
        assert len(wd.load_context_bias()) == MAX_CONTEXT_BIAS_TERMS

    def test_clear(self, tmp_path: Path):
        wd = WorkDirectory(tmp_path)
        wd.save_context_bias(["test"])
        assert wd.context_bias_file.exists()
        wd.clear_context_bias()
        assert not wd.context_bias_file.exists()
        assert wd.load_context_bias() == []

    def test_load_no_file(self, tmp_path: Path):
        wd = WorkDirectory(tmp_path)
        assert wd.load_context_bias() == []

    def test_strips_whitespace(self, tmp_path: Path):
        wd = WorkDirectory(tmp_path)
        wd.save_context_bias(["  ORCA  ", "", "  DFT  ", "  "])
        loaded = wd.load_context_bias()
        assert loaded == ["ORCA", "DFT"]
