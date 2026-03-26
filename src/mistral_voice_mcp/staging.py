"""Staging session management for audio transcription workflow."""

import re
import shutil
import uuid
from pathlib import Path

from mistral_voice_mcp.workdir import AUDIO_EXTENSIONS


def slugify(name: str, max_length: int = 80) -> str:
    """Convert a representative name to a filesystem-safe slug.

    Examples:
        "ChromSword vs DryLab Method Optimization" -> "chromsword-vs-drylab-method-optimization"
        "ORCA Software: DFT Calculations!" -> "orca-software-dft-calculations"
    """
    slug = name.lower()
    slug = re.sub(r"[_\s]+", "-", slug)
    slug = re.sub(r"[^a-z0-9\-]", "", slug)
    slug = re.sub(r"-{2,}", "-", slug)
    slug = slug.strip("-")
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("-")
    return slug


class StagingSession:
    """A staging session holds audio files and their transcription artifacts.

    Created under ``<workdir>/.staging/<session_id>/``.
    """

    def __init__(self, workdir_root: Path, session_id: str | None = None) -> None:
        self._workdir_root = Path(workdir_root).resolve()
        self._session_id = session_id or uuid.uuid4().hex[:8]
        self._session_dir = self._workdir_root / ".staging" / self._session_id
        self._session_dir.mkdir(parents=True, exist_ok=True)

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def session_dir(self) -> Path:
        return self._session_dir

    @property
    def audio_files(self) -> list[Path]:
        """Sorted list of audio files in the session directory."""
        return sorted(
            p for p in self._session_dir.iterdir()
            if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        )

    @property
    def transcript_files(self) -> list[Path]:
        """Sorted list of .md transcript files in the session directory."""
        return sorted(
            p for p in self._session_dir.iterdir()
            if p.is_file() and p.suffix == ".md"
        )

    def is_fully_transcribed(self) -> bool:
        """Check if every audio file has a corresponding .md transcript."""
        audio_stems = {p.stem for p in self.audio_files}
        transcript_stems = {p.stem for p in self.transcript_files}
        return bool(audio_stems) and audio_stems <= transcript_stems

    def merge_transcripts(self) -> str:
        """Merge all transcript .md files into a single markdown string.

        Single file: returns content as-is (no Part header).
        Multiple files: concatenates with ``# Part: <filename>`` H1 headers,
        ordered alphabetically by audio filename.
        """
        transcripts = self.transcript_files
        if not transcripts:
            return ""

        if len(transcripts) == 1:
            return transcripts[0].read_text()

        parts: list[str] = []
        for md_file in transcripts:
            # Find the matching audio file to get the original filename
            audio_name = None
            for audio in self.audio_files:
                if audio.stem == md_file.stem:
                    audio_name = audio.name
                    break
            header = f"# Part: {audio_name or md_file.stem}"
            content = md_file.read_text().strip()
            parts.append(f"{header}\n\n{content}")

        return "\n\n".join(parts) + "\n"

    def finalize(self, name: str) -> Path:
        """Move session contents into a named directory under workdir root.

        Creates ``<workdir>/<slug>/``, moves audio files, writes merged
        ``transcript.md``, and removes the staging session directory.

        Returns the path to the finalized recording directory.
        """
        slug = slugify(name)
        if not slug:
            raise ValueError("Name cannot be empty or produce an empty slug.")

        final_dir = self._workdir_root / slug
        if final_dir.exists():
            raise ValueError(
                f"Directory already exists: {final_dir.name}. Choose a different name."
            )

        final_dir.mkdir(parents=True)

        # Merge transcripts before moving files (needs both audio + md present)
        merged = self.merge_transcripts()

        # Move audio files
        for audio in self.audio_files:
            shutil.move(str(audio), str(final_dir / audio.name))

        # Write merged transcript
        if merged:
            (final_dir / "transcript.md").write_text(merged)

        # Clean up staging directory
        shutil.rmtree(self._session_dir)

        return final_dir


def create_session(workdir_root: Path, audio_paths: list[Path]) -> StagingSession:
    """Create a new staging session and copy audio files into it."""
    if not audio_paths:
        raise ValueError("At least one audio file is required.")

    session = StagingSession(workdir_root)
    for src in audio_paths:
        if not src.exists():
            raise FileNotFoundError(f"Audio file not found: {src}")
        shutil.copy2(str(src), str(session.session_dir / src.name))

    return session


def list_sessions(workdir_root: Path) -> list[StagingSession]:
    """List all active staging sessions."""
    staging_dir = Path(workdir_root).resolve() / ".staging"
    if not staging_dir.exists():
        return []
    sessions = []
    for d in sorted(staging_dir.iterdir()):
        if d.is_dir():
            sessions.append(StagingSession(workdir_root, d.name))
    return sessions


def get_session(workdir_root: Path, session_id: str) -> StagingSession:
    """Retrieve an existing staging session by ID."""
    session_dir = Path(workdir_root).resolve() / ".staging" / session_id
    if not session_dir.exists():
        raise ValueError(f"Staging session not found: {session_id}")
    return StagingSession(workdir_root, session_id)
