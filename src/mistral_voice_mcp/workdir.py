"""Work directory management for audio transcription."""

from pathlib import Path

AUDIO_EXTENSIONS: set[str] = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mpeg"}
MAX_CONTEXT_BIAS_TERMS: int = 100

# Directories that are part of the internal structure, not recordings
_RESERVED_DIRS: set[str] = {"inbox", ".staging"}


class WorkDirectory:
    """Manages a work directory with inbox/, .staging/, and recording directories.

    Users drop audio files in inbox/. Transcription happens in .staging/.
    Finalized recordings live as named directories under the root.
    """

    def __init__(self, path: str | Path) -> None:
        self._root = Path(path).resolve()
        if not self._root.exists():
            self._root.mkdir(parents=True)
        self.inbox_dir.mkdir(exist_ok=True)
        self.staging_dir.mkdir(exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def inbox_dir(self) -> Path:
        return self._root / "inbox"

    @property
    def staging_dir(self) -> Path:
        return self._root / ".staging"

    @property
    def context_bias_file(self) -> Path:
        return self._root / "context_bias.txt"

    def scan_inbox(self) -> list[Path]:
        """Return sorted list of audio files in inbox/ (recursive)."""
        files = [
            p
            for p in self.inbox_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        ]
        return sorted(files)

    def scan_recordings(self) -> list[Path]:
        """Return sorted list of finalized recording directories under root.

        Excludes reserved directories (inbox, .staging) and hidden directories.
        """
        return sorted(
            d
            for d in self._root.iterdir()
            if d.is_dir() and d.name not in _RESERVED_DIRS and not d.name.startswith(".")
        )

    def status(self) -> dict[str, int]:
        """Return counts of inbox files, active staging sessions, and recordings."""
        inbox = len(self.scan_inbox())
        staging = len([
            d for d in self.staging_dir.iterdir() if d.is_dir()
        ]) if self.staging_dir.exists() else 0
        recordings = len(self.scan_recordings())
        return {
            "inbox": inbox,
            "staging": staging,
            "recordings": recordings,
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
