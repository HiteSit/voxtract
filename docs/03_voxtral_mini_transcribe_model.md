# Voxtral Mini Transcribe 2 - Model Reference

> Source: https://docs.mistral.ai/models/voxtral-mini-transcribe-26-02

## Model Identifiers

| Field | Value |
|-------|-------|
| Primary ID | `voxtral-mini-2602` |
| Latest alias | `voxtral-mini-latest` |
| Tier | Premier ("cutting edge of technology for enterprise use") |
| Release date | February 4, 2026 |
| Pricing | **$0.003 per minute** of audio |

---

## Description

An efficient audio input model, fine-tuned and optimized for **transcription purposes only**.

---

## Capabilities

- Audio transcription (core function)
- Transcription with timestamps (segment and word level)
- Chat completions integration
- Function calling support
- Structured outputs
- Batch inference processing

---

## API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/v1/audio/transcriptions` | Standard transcription |
| `/v1/audio/transcriptions` (with timestamps) | Transcription + timestamp generation |

---

## Integration Points

- Agent conversations (`/v1/agents`)
- Built-in tools framework
- Moderation systems (`/v1/moderations`)
- Embedding capabilities (`/v1/embeddings`)
- Document AI features (OCR, annotations, BBox extraction)

---

## Quick Reference

```python
from mistralai.client import Mistral

client = Mistral(api_key="YOUR_API_KEY")

# Basic transcription
with open("audio.mp3", "rb") as f:
    result = client.audio.transcriptions.complete(
        model="voxtral-mini-2602",
        file={"content": f, "file_name": "audio.mp3"}
    )

# With timestamps
result = client.audio.transcriptions.complete(
    model="voxtral-mini-2602",
    file={"content": f, "file_name": "audio.mp3"},
    timestamp_granularities=["word"]
)

# With diarization
result = client.audio.transcriptions.complete(
    model="voxtral-mini-2602",
    file={"content": f, "file_name": "audio.mp3"},
    diarize=True
)

# With context biasing
result = client.audio.transcriptions.complete(
    model="voxtral-mini-2602",
    file={"content": f, "file_name": "audio.mp3"},
    context_bias="ProperNoun1,TechnicalTerm2"
)
```

---

## Governance

Legal/governance documentation: https://legal.mistral.ai/ai-governance/models/voxtral-small
