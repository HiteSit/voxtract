"""Tests for WorkDirectory class."""

from pathlib import Path

import pytest

from mistral_voice_mcp.workdir import MAX_CONTEXT_BIAS_TERMS, WorkDirectory


class TestInit:
    def test_creates_subdirs(self, tmp_path: Path):
        wd = WorkDirectory(tmp_path / "new_project")
        assert wd.input_dir.is_dir()
        assert wd.output_dir.is_dir()

    def test_existing_dirs_no_error(self, workdir: Path):
        wd = WorkDirectory(workdir)
        assert wd.root == workdir


class TestScanInputs:
    def test_empty(self, workdir: Path):
        wd = WorkDirectory(workdir)
        assert wd.scan_inputs() == []

    def test_filters_by_extension(self, workdir: Path):
        (workdir / "input" / "audio.mp3").write_bytes(b"\x00")
        (workdir / "input" / "notes.txt").write_text("hello")
        (workdir / "input" / "data.csv").write_text("a,b")
        wd = WorkDirectory(workdir)
        results = wd.scan_inputs()
        assert len(results) == 1
        assert results[0].name == "audio.mp3"

    def test_recursive(self, workdir: Path):
        sub = workdir / "input" / "session1"
        sub.mkdir()
        (sub / "recording.wav").write_bytes(b"\x00")
        (workdir / "input" / "main.flac").write_bytes(b"\x00")
        wd = WorkDirectory(workdir)
        results = wd.scan_inputs()
        assert len(results) == 2

    def test_all_extensions(self, workdir: Path):
        for ext in [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mpeg"]:
            (workdir / "input" / f"file{ext}").write_bytes(b"\x00")
        wd = WorkDirectory(workdir)
        assert len(wd.scan_inputs()) == 6


class TestOutputPathMirroring:
    def test_mirrors_structure(self, workdir: Path):
        sub = workdir / "input" / "batch" / "sub"
        sub.mkdir(parents=True)
        audio = sub / "call.mp3"
        audio.write_bytes(b"\x00")

        wd = WorkDirectory(workdir)
        out = wd.get_output_path(audio)
        assert out == workdir / "output" / "batch" / "sub" / "call.json"

    def test_creates_parent_dirs(self, workdir: Path):
        sub = workdir / "input" / "deep" / "nested"
        sub.mkdir(parents=True)
        audio = sub / "file.mp3"
        audio.write_bytes(b"\x00")

        wd = WorkDirectory(workdir)
        out = wd.get_output_path(audio)
        assert out.parent.is_dir()

    def test_custom_suffix(self, workdir: Path):
        audio = workdir / "input" / "file.mp3"
        audio.write_bytes(b"\x00")
        wd = WorkDirectory(workdir)
        out = wd.get_output_path(audio, suffix=".txt")
        assert out.suffix == ".txt"


class TestTranscriptionStatus:
    def test_is_transcribed_false(self, workdir: Path):
        audio = workdir / "input" / "file.mp3"
        audio.write_bytes(b"\x00")
        wd = WorkDirectory(workdir)
        assert not wd.is_transcribed(audio)

    def test_is_transcribed_true(self, workdir: Path):
        audio = workdir / "input" / "file.mp3"
        audio.write_bytes(b"\x00")
        (workdir / "output" / "file.json").write_text("{}")
        wd = WorkDirectory(workdir)
        assert wd.is_transcribed(audio)

    def test_pending_files(self, workdir: Path):
        (workdir / "input" / "done.mp3").write_bytes(b"\x00")
        (workdir / "input" / "pending.wav").write_bytes(b"\x00")
        (workdir / "output" / "done.json").write_text("{}")
        wd = WorkDirectory(workdir)
        pending = wd.pending_files()
        assert len(pending) == 1
        assert pending[0].name == "pending.wav"

    def test_status_counts(self, workdir: Path):
        (workdir / "input" / "a.mp3").write_bytes(b"\x00")
        (workdir / "input" / "b.mp3").write_bytes(b"\x00")
        (workdir / "input" / "c.mp3").write_bytes(b"\x00")
        (workdir / "output" / "a.json").write_text("{}")
        wd = WorkDirectory(workdir)
        s = wd.status()
        assert s == {"total_inputs": 3, "transcribed": 1, "pending": 2}


class TestContextBias:
    def test_save_and_load_roundtrip(self, workdir: Path):
        wd = WorkDirectory(workdir)
        terms = ["ORCA", "DFT", "MCR reaction"]
        wd.save_context_bias(terms)
        loaded = wd.load_context_bias()
        assert loaded == terms

    def test_max_terms_exceeded(self, workdir: Path):
        wd = WorkDirectory(workdir)
        terms = [f"term_{i}" for i in range(MAX_CONTEXT_BIAS_TERMS + 1)]
        with pytest.raises(ValueError, match="at most"):
            wd.save_context_bias(terms)

    def test_max_terms_exact_limit(self, workdir: Path):
        wd = WorkDirectory(workdir)
        terms = [f"term_{i}" for i in range(MAX_CONTEXT_BIAS_TERMS)]
        wd.save_context_bias(terms)
        assert len(wd.load_context_bias()) == MAX_CONTEXT_BIAS_TERMS

    def test_clear(self, workdir: Path):
        wd = WorkDirectory(workdir)
        wd.save_context_bias(["test"])
        assert wd.context_bias_file.exists()
        wd.clear_context_bias()
        assert not wd.context_bias_file.exists()
        assert wd.load_context_bias() == []

    def test_load_no_file(self, workdir: Path):
        wd = WorkDirectory(workdir)
        assert wd.load_context_bias() == []

    def test_strips_whitespace(self, workdir: Path):
        wd = WorkDirectory(workdir)
        wd.save_context_bias(["  ORCA  ", "", "  DFT  ", "  "])
        loaded = wd.load_context_bias()
        assert loaded == ["ORCA", "DFT"]
