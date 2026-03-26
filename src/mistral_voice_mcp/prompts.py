"""Prompt templates for post-processing transcriptions."""


def clean_transcript_messages(transcript: str) -> list[dict]:
    """Return messages for cleaning a raw transcript."""
    return [
        {
            "role": "user",
            "content": (
                "Clean up the following raw audio transcript. "
                "Remove filler words (um, uh, like, you know, so, basically), "
                "fix grammar and punctuation, and break into proper paragraphs. "
                "Preserve the original meaning, tone, and speaker voice. "
                "Do not add information that is not in the original. "
                "If there are speaker labels, keep them.\n\n"
                "--- RAW TRANSCRIPT ---\n"
                f"{transcript}\n"
                "--- END ---"
            ),
        }
    ]


def meeting_notes_messages(transcript: str) -> list[dict]:
    """Return messages for structuring a transcript into meeting notes."""
    return [
        {
            "role": "user",
            "content": (
                "Convert the following transcript into structured meeting notes. "
                "Include these sections:\n"
                "1. **Attendees/Speakers** - list identified speakers\n"
                "2. **Topics Discussed** - organized by subject\n"
                "3. **Decisions Made** - any conclusions reached\n"
                "4. **Action Items** - tasks with owners if identifiable\n"
                "5. **Open Questions** - unresolved points\n\n"
                "Be concise but capture all substantive points.\n\n"
                "--- TRANSCRIPT ---\n"
                f"{transcript}\n"
                "--- END ---"
            ),
        }
    ]


def summarize_transcript_messages(transcript: str) -> list[dict]:
    """Return messages for summarizing a transcript."""
    return [
        {
            "role": "user",
            "content": (
                "Provide a concise summary of the following transcript. "
                "Organize key points by topic. "
                "Include notable quotes if they add value. "
                "Keep the summary to roughly 20% of the original length.\n\n"
                "--- TRANSCRIPT ---\n"
                f"{transcript}\n"
                "--- END ---"
            ),
        }
    ]


def technical_notes_messages(transcript: str) -> list[dict]:
    """Return messages for extracting technical content into structured notes."""
    return [
        {
            "role": "user",
            "content": (
                "Extract technical content from the following transcript "
                "and organize into structured notes:\n"
                "1. **Technical Terms & Definitions** - key terminology mentioned\n"
                "2. **Tools & Software** - any software, libraries, or platforms referenced\n"
                "3. **Concepts & Methods** - scientific or technical methodologies discussed\n"
                "4. **Key Ideas & Proposals** - main technical ideas or proposals\n"
                "5. **References** - any papers, resources, or links mentioned\n\n"
                "Be precise with technical terminology.\n\n"
                "--- TRANSCRIPT ---\n"
                f"{transcript}\n"
                "--- END ---"
            ),
        }
    ]
