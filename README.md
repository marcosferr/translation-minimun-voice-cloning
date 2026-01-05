
# Translation Pipeline with Voice Cloning

> Fully automated video translation with voice cloning.  
> Inspired by how Meta translates Reels at scale.  
> **Only 5 seconds of voice required.**

This repository provides a **minimal, end-to-end pipeline** for translating videos while **preserving the original speaker’s voice**, using modern speech-to-text, translation, and TTS models.

The main goal is to make **high-quality English technical content accessible to Spanish-speaking audiences** (and other languages) with **minimal setup and zero manual steps**.

---

## ✨ Why this project?

Most valuable technical content is published in English.  
Language should not be the limiting factor for learning.

This project demonstrates how modern AI tooling can reduce that barrier **without sacrificing speaker identity**, using an approach similar to large-scale localization systems used by platforms like **Meta**.

---

## 🔑 Key Features

- ✅ Fully automated pipeline (no manual editing)
- 🎙 Voice cloning with **only 3–10 seconds of clean audio**
- 🧩 Segment-by-segment translation optimized for lip-sync
- ⚙️ Simple, reproducible setup
- 🧪 Designed to be extended or integrated into larger workflows

---

## 🧠 Pipeline Overview



FFmpeg → Whisper API → GPT-5 mini → Chatterbox TTS → FFmpeg

````

| Step | Description |
|-----|------------|
| FFmpeg | Extracts and merges audio/video |
| Whisper API | Transcribes audio with timestamps |
| GPT-5 mini | Translates segments individually |
| Chatterbox TTS | Generates cloned voice |
| FFmpeg | Assembles final translated video |

---

## 🚀 Quick Start

### Requirements

- **Python 3.10+**
- **FFmpeg** (installed and in PATH)
- Optional: CUDA-capable GPU (CPU works, slower)

Verify FFmpeg:
```bash
ffmpeg -version
````

---

### Installation

```bash
git clone https://github.com/marcosferr/translation-minimun-voice-cloning.git
cd translation-minimun-voice-cloning

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

---

### OpenAI API Key

Create a `.env` file:

```bash
cp .env.example .env
```

Add your key:

```env
OPENAI_API_KEY=sk-your-actual-api-key-here
```

---

## ▶️ Usage

### Basic example

```bash
python translate.py input.mp4 voice.wav output.mp4
```

### Custom languages

```bash
python translate.py input.mp4 voice.wav output.mp4 \
  --source-lang en \
  --target-lang es
```

### Force CPU

```bash
python translate.py input.mp4 voice.wav output.mp4 --device cpu
```

---

## 📥 Input Requirements

### Video

* Format: MP4, AVI, MOV (FFmpeg compatible)
* Audio extracted automatically

### Voice Prompt

* Format: WAV (recommended)
* Duration: **3–10 seconds**
* Clean audio, single speaker, no background noise

---

## 🌍 Supported Languages

| Language   | Code |
| ---------- | ---- |
| Spanish    | es   |
| English    | en   |
| French     | fr   |
| German     | de   |
| Italian    | it   |
| Portuguese | pt   |
| Japanese   | ja   |
| Korean     | ko   |
| Chinese    | zh   |

Full list:
[https://platform.openai.com/docs/guides/speech-to-text/supported-languages](https://platform.openai.com/docs/guides/speech-to-text/supported-languages)

---

## 🧪 Reproducible Test Setup

* OS: Windows
* Python: 3.11
* GPU tested: NVIDIA RTX 4060 (CUDA 12.4)

Example:

```bash
conda create -n tts-env python=3.11 -y
conda activate tts-env
conda install -c conda-forge ffmpeg numpy<1.26 pysoundfile -y
pip install -r requirements.txt
pip install chatterbox-tts python-dotenv openai
```

---

## ⚠️ Current Limitations

This minimal implementation does **not** handle:

* Time-stretching to match original segment duration
* Silence preservation
* Retry logic
* Batch/queue processing
* Manual post-editing

---

## 🛣 Roadmap / Next Steps

* ⏱ Time-stretching for better sync
* 🔇 Silence preservation
* 🧱 Modular pipeline stages
* 🧪 CLI improvements (progress, verbose mode)
* 👄 Wav2Lip integration for lip-sync

---

## 🙌 Credits

Video reference used for testing:
**Stephane Maarek — Amazon GuardDuty Deep Dive**

---

## 📄 License

MIT License — free to use, modify, and extend.


