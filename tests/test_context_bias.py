"""Tests for context bias parsing and validation."""

from pathlib import Path

import pytest

from mistral_voice_mcp.workdir import WorkDirectory


class TestContextBiasParsing:
    """Test various input formats for context bias terms."""

    def test_comma_separated(self, workdir: Path):
        wd = WorkDirectory(workdir)
        terms = "ORCA, DFT, MCR reaction".split(",")
        terms = [t.strip() for t in terms]
        wd.save_context_bias(terms)
        assert wd.load_context_bias() == ["ORCA", "DFT", "MCR reaction"]

    def test_newline_separated(self, workdir: Path):
        wd = WorkDirectory(workdir)
        terms = ["ORCA", "DFT", "MCR reaction"]
        wd.save_context_bias(terms)
        loaded = wd.load_context_bias()
        assert loaded == ["ORCA", "DFT", "MCR reaction"]

    def test_strips_whitespace(self, workdir: Path):
        wd = WorkDirectory(workdir)
        terms = ["  ORCA  ", "  DFT  ", "  MCR  "]
        wd.save_context_bias(terms)
        loaded = wd.load_context_bias()
        assert loaded == ["ORCA", "DFT", "MCR"]

    def test_empty_terms_filtered(self, workdir: Path):
        wd = WorkDirectory(workdir)
        terms = ["ORCA", "", "  ", "DFT", ""]
        wd.save_context_bias(terms)
        loaded = wd.load_context_bias()
        assert loaded == ["ORCA", "DFT"]

    def test_from_file(self, workdir: Path):
        """Simulate reading terms from a text file."""
        wd = WorkDirectory(workdir)
        bias_file = wd.input_dir / "my_terms.txt"
        bias_file.write_text("ORCA\nDFT\ntelazo piperazine\nUgi reaction\n")

        raw = bias_file.read_text()
        terms = [t.strip() for t in raw.splitlines() if t.strip()]
        wd.save_context_bias(terms)

        loaded = wd.load_context_bias()
        assert loaded == ["ORCA", "DFT", "telazo piperazine", "Ugi reaction"]
