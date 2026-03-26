"""Prompt templates for post-processing transcriptions."""


def clean_transcript_messages(transcript: str) -> list[dict]:
    """Return messages for cleaning and restructuring a raw transcript.

    The prompt instructs the LLM to fix logic flow, remove filler words,
    improve coherence, and return clean markdown preserving the speaker
    heading structure.
    """
    return [
        {
            "role": "user",
            "content": (
                "You are editing a raw audio transcript that has been automatically "
                "transcribed and formatted as markdown with speaker headings "
                "(## Speaker N). The transcript may be confusing, jump between "
                "topics, contain filler words, or have unclear logic.\n\n"
                "Your task:\n"
                "1. Remove filler words (um, uh, like, you know, so, basically, "
                "I mean, sort of, kind of)\n"
                "2. Fix grammar, punctuation, and sentence structure\n"
                "3. Improve logical flow — reorder or restructure sentences within "
                "each speaker block so the ideas progress clearly\n"
                "4. Break long monologues into coherent paragraphs\n"
                "5. Preserve the original meaning, intent, and all substantive content\n"
                "6. Keep the ## Speaker N markdown headings exactly as they are\n"
                "7. Do NOT add information that is not in the original\n"
                "8. Do NOT merge or reorder speaker blocks — keep the conversation "
                "sequence intact\n\n"
                "Return ONLY the cleaned markdown transcript, nothing else.\n\n"
                "--- RAW TRANSCRIPT ---\n"
                f"{transcript}\n"
                "--- END ---"
            ),
        }
    ]
