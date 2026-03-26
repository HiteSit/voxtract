"""Work directory management for audio transcription."""

from pathlib import Path

AUDIO_EXTENSIONS: set[str] = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mpeg"}
MAX_CONTEXT_BIAS_TERMS: int = 100


class WorkDirectory:
    """Manages a work directory with input/ and output/ subdirectories.

    All audio files go in input/, transcription results in output/.
    The output structure mirrors the input structure.
    """

    def __init__(self, path: str | Path) -> None:
        self._root = Path(path).resolve()
        if not self._root.exists():
            self._root.mkdir(parents=True)
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def input_dir(self) -> Path:
        return self._root / "input"

    @property
    def output_dir(self) -> Path:
        return self._root / "output"

    @property
    def context_bias_file(self) -> Path:
        return self._root / "context_bias.txt"

    def scan_inputs(self) -> list[Path]:
        """Return sorted list of audio files in input/ (recursive)."""
        files = [
            p
            for p in self.input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        ]
        return sorted(files)

    def get_output_path(self, input_path: Path, suffix: str = ".json") -> Path:
        """Mirror input_path into output/, changing the file extension.

        Creates intermediate output subdirectories as needed.
        """
        relative = input_path.relative_to(self.input_dir)
        output_path = self.output_dir / relative.with_suffix(suffix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def is_transcribed(self, input_path: Path) -> bool:
        """Check if a .json transcription exists for this input file."""
        json_path = self.get_output_path(input_path, suffix=".json")
        return json_path.exists()

    def pending_files(self) -> list[Path]:
        """Return input files that have no corresponding .json in output/."""
        return [f for f in self.scan_inputs() if not self.is_transcribed(f)]

    def status(self) -> dict[str, int]:
        """Return counts of total inputs, transcribed, and pending files."""
        all_inputs = self.scan_inputs()
        transcribed = sum(1 for f in all_inputs if self.is_transcribed(f))
        return {
            "total_inputs": len(all_inputs),
            "transcribed": transcribed,
            "pending": len(all_inputs) - transcribed,
        }

    def load_context_bias(self) -> list[str]:
        """Read context_bias.txt and return list of terms. Empty list if missing."""
        if not self.context_bias_file.exists():
            return []
        text = self.context_bias_file.read_text().strip()
        if not text:
            return []
        return [term.strip() for term in text.splitlines() if term.strip()]

    def save_context_bias(self, terms: list[str]) -> None:
        """Write terms to context_bias.txt, one per line. Max 100 terms."""
        clean = [t.strip() for t in terms if t.strip()]
        if len(clean) > MAX_CONTEXT_BIAS_TERMS:
            raise ValueError(
                f"Context bias supports at most {MAX_CONTEXT_BIAS_TERMS} terms, "
                f"got {len(clean)}"
            )
        self.context_bias_file.write_text("\n".join(clean) + "\n")

    def clear_context_bias(self) -> None:
        """Delete context_bias.txt if it exists."""
        if self.context_bias_file.exists():
            self.context_bias_file.unlink()
