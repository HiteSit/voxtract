# Voxtral Transcribe 2 - Announcement & Technical Details

> Source: https://mistral.ai/news/voxtral-transcribe-2

## Overview

Voxtral Transcribe 2 comprises two speech-to-text models:

- **Voxtral Mini Transcribe V2** - Batch transcription
- **Voxtral Realtime** - Live/streaming transcription

---

## Model Architecture

### Voxtral Mini Transcribe V2

- Batch processing model
- Supports up to **3 hours of audio** per request
- Significantly improved transcription and diarization quality over V1

### Voxtral Realtime

- Purpose-built streaming architecture (transcribes audio as it arrives)
- Configurable latency down to **sub-200ms**
- **4B parameter** footprint (suitable for edge device deployment)
- Natively multilingual across 13 languages
- Released under **Apache 2.0** open-weights license on Hugging Face

---

## Supported Languages (Both Models)

English, Chinese, Hindi, Spanish, Arabic, French, Portuguese, Russian, German, Japanese, Korean, Italian, and Dutch.

**13 languages total.**

---

## Key Features

### Speaker Diarization

- Provides speaker labels and timestamps
- With overlapping speech, the model typically transcribes one speaker

### Context Biasing

- Provide up to **100 words or phrases** to guide correct spellings
- Useful for names, technical terms, domain-specific vocabulary
- Optimized for English; experimental support for other languages

### Word-Level Timestamps

- Enables subtitle generation, audio search, and content alignment

### Noise Robustness

- Maintains accuracy in challenging acoustic environments
- Tested on factory floors and call centers

---

## Performance Benchmarks

### Word Error Rate (FLEURS benchmark)

- Voxtral Mini Transcribe V2 achieves approximately **4% WER** across top-10 languages
- Outperforms: GPT-4o mini Transcribe, Gemini 2.5 Flash, Assembly Universal, Deepgram Nova

### Latency (Voxtral Realtime)

- At 2.4s delay (subtitling use case): matches Mini Transcribe V2 quality
- At 480ms delay: maintains 1-2% word error rate

### Diarization Error Rate

Tested across:
- English: Switchboard, CallHome, AMI-IHM, AMI-SDM, SBCSAE
- Multilingual: TalkBank data (German, Spanish, English, Chinese, Japanese)

---

## Pricing

| Model | Price |
|-------|-------|
| Voxtral Mini Transcribe V2 | **$0.003 per minute** |
| Voxtral Realtime | **$0.006 per minute** |

- ~3x faster than ElevenLabs Scribe v2
- 1/5th the cost of ElevenLabs at matching quality

---

## Supported Audio Formats

`.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg` - up to **1GB** each.

---

## Use Cases

- **Meeting Intelligence:** Multilingual transcription with speaker attribution
- **Voice Agents:** Sub-200ms latency for natural voice interfaces
- **Contact Center Automation:** Real-time transcription with sentiment analysis
- **Media/Broadcast:** Live multilingual subtitles
- **Compliance:** GDPR and HIPAA-compliant deployments (on-premise / private cloud)

---

## Availability

- Voxtral Mini Transcribe V2: Available via API
- Voxtral Realtime: Available via API + open-weights on Hugging Face
- Both accessible through Mistral Studio playground and Le Chat
