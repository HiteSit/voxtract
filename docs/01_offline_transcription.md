# Offline Audio Transcription with Mistral

> Source: https://docs.mistral.ai/capabilities/audio_transcription/offline_transcription

## Audio-Capable Models

| Model | ID | Use Case |
|-------|-----|----------|
| Voxtral Small | `voxtral-small-latest` | Chat with audio input |
| Voxtral Mini | `voxtral-mini-latest` | Chat with audio input |
| Voxtral Mini Transcribe | `voxtral-mini-latest` (via `audio/transcriptions`) | Transcription-only |

**Model version mapping:**

- Chat endpoint: `voxtral-mini-latest` resolves to `voxtral-mini-2507`
- Transcription endpoint: `voxtral-mini-latest` resolves to `voxtral-mini-2602`

---

## Chat with Audio (Base64 Encoding)

```python
import base64
from mistralai.client import Mistral

client = Mistral(api_key=api_key)
with open("examples/files/bcn_weather.mp3", "rb") as f:
    content = f.read()
audio_base64 = base64.b64encode(content).decode('utf-8')

chat_response = client.chat.complete(
    model="voxtral-mini-latest",
    messages=[{
        "role": "user",
        "content": [
            {"type": "input_audio", "input_audio": audio_base64},
            {"type": "text", "text": "What's in this file?"}
        ]
    }]
)
```

---

## Transcription Endpoint

### Basic Transcription

```python
from mistralai.client import Mistral

client = Mistral(api_key=api_key)
with open("/path/to/file/audio.mp3", "rb") as f:
    transcription_response = client.audio.transcriptions.complete(
        model="voxtral-mini-latest",
        file={"content": f, "file_name": "audio.mp3"}
    )
```

### Transcription with Timestamps

```python
transcription_response = client.audio.transcriptions.complete(
    model="voxtral-mini-latest",
    file_url="https://docs.mistral.ai/audio/obama.mp3",
    timestamp_granularities=["segment"]  # or "word"
)
```

### Context Biasing

Provide up to 100 words/phrases to guide spelling of proper nouns, technical terms, or domain-specific vocabulary.

```python
transcription_response = client.audio.transcriptions.complete(
    model="voxtral-mini-2602",
    file_url="https://docs.mistral.ai/audio/obama.mp3",
    context_bias="Chicago,Joplin,Boston,Charleston,farewell_address"
)
```

---

## Key Parameters

| Parameter | Purpose | Notes |
|-----------|---------|-------|
| `model` | Model to use | `voxtral-mini-latest` or `voxtral-mini-2602` |
| `file` | Local file upload | Dict with `content` (file obj) and `file_name` |
| `file_url` | Remote file URL | Alternative to `file` |
| `timestamp_granularities` | Timestamp level | `["segment"]` or `["word"]` |
| `diarize` | Speaker identification | Boolean |
| `context_bias` | Spelling guidance | Comma-separated string, up to 100 words |
| `language` | Manual language spec | Improves accuracy for known language |

**Important:** `timestamp_granularities` and `language` are mutually exclusive.

---

## Tips

- For faster transcription, upload audio files rather than using URLs.
- Consult the Chat Completions API reference for full parameter details.
