# Translation Pipeline with Voice Cloning

Minimal translation pipeline using Whisper API, GPT-5 mini, and Chatterbox TTS for voice cloning.

## Pipeline Overview

```
FFmpeg -> Whisper API -> GPT-5 mini -> Chatterbox TTS -> FFmpeg
```

1. **FFmpeg**: Extract audio from video
2. **Whisper API**: Transcribe with timestamps
3. **GPT-5 mini**: Translate segment by segment
4. **Chatterbox TTS**: Generate cloned voice
5. **FFmpeg**: Merge audio with video

## Prerequisites

### System Requirements

- **FFmpeg** installed and in PATH
  - Download: https://ffmpeg.org/download.html
  - Verify: `ffmpeg -version`

### Hardware

- **GPU**: CUDA-capable GPU recommended (optional, CPU works but slower)
- **RAM**: 8GB+ recommended

### Software

- Python 3.10+

## Installation

### 1. Clone or download this project

```bash
cd translation-minimun-voice-cloning
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv

# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up OpenAI API key

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

Get your API key from: https://platform.openai.com/api-keys

## Usage

### Basic Usage

```bash
python translate.py input.mp4 voice_prompt.wav output.mp4
```

### With Custom Languages

```bash
python translate.py input.mp4 voice_prompt.wav output.mp4 \
    --source-lang es \
    --target-lang en
```

### Force CPU (if no GPU)

```bash
python translate.py input.mp4 voice_prompt.wav output.mp4 --device cpu
```

### All Options

```
positional arguments:
  video_input          Input video file (e.g., input.mp4)
  voice_prompt         Voice reference audio (e.g., voice.wav)
  video_output         Output video file (e.g., output.mp4)

options:
  --source-lang        Source language code (default: es)
  --target-lang        Target language code (default: en)
  --device             Device for Chatterbox TTS: cuda or cpu (default: cuda)
  --api-key            OpenAI API key (or use OPENAI_API_KEY env var)
```

## Input Files

### Video Input

- Format: MP4, AVI, MOV, etc. (FFmpeg compatible)
- Audio: Will be extracted automatically

### Voice Prompt

- Format: WAV recommended
- Duration: 3-10 seconds
- Quality: Clean, clear voice sample
- Content: The target voice you want to clone

## Language Codes

Common language codes:

| Language | Code |
|----------|------|
| Spanish  | `es` |
| English  | `en` |
| French   | `fr` |
| German   | `de` |
| Italian  | `it` |
| Portuguese | `pt` |
| Japanese | `ja` |
| Korean   | `ko` |
| Chinese  | `zh` |

Full list: https://platform.openai.com/docs/guides/speech-to-text/supported-languages

## Example Workflow

```bash
# 1. Prepare your files
# - input.mp4: Source video
# - voice.wav: Clean voice sample (3-10 seconds)

# 2. Run translation (Spanish to English)
python translate.py input.mp4 voice.wav output.mp4 --source-lang es --target-lang en

# 3. Result
# - output.mp4: Translated video with cloned voice

---

## Reproducible setup used for testing ✅

- **OS**: Windows (tested here on Windows with a user-installed FFmpeg)
- **Conda environment**: `tts-env` with **Python 3.11.14`

  Example commands used to reproduce the environment:
  ```bash
  conda create -n tts-env python=3.11 -y
  conda activate tts-env

  # Install system/binary deps with conda for Windows
  conda install -c conda-forge ffmpeg "numpy>=1.24.0,<1.26.0" pysoundfile -y

  # Install Python packages from requirements and extras
  pip install -r requirements.txt
  pip install chatterbox-tts python-dotenv openai

  # Install CUDA-enabled PyTorch (for RTX 40-series tested here with CUDA 12.4)
  pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio --force-reinstall
  ```

- **GPU used in testing**: NVIDIA RTX 4060 (6 GB VRAM) — PyTorch built with CUDA 12.4 was installed and verified with `torch.cuda.is_available()`.
- **FFmpeg**: script checks PATH first and falls back to a hardcoded Windows install path in the test machine. Example hardcoded path used in the script:

  `C:\Users\ferre\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe`

- **Notes on dependency tweaks**: During setup I installed a CUDA-enabled torch wheel and then pinned back `numpy<1.26.0` and `pillow<12.0` to satisfy `chatterbox-tts` / `gradio` constraints.

- **Command used to run the example in this repo (what I ran):**
  ```bash
  conda activate tts-env && python translate.py input.mp4 output.mp4 --source-lang en --target-lang es
  ```

- **Files used in this test**:
  - `input.mp4` (source video included in repo)
  - `output.mp4` (generated result)

---

## Credits / Video Reference

Guide credit to: [Stephane Maarek — Amazon GuardDuty Deep Dive](https://youtu.be/M4aOKikd7-s?si=3uEMYeqntCBEc_9D)

---

## How It Works
```

## How It Works

### Step 1: Audio Extraction
FFmpeg extracts mono audio at 16kHz from the input video.

### Step 2: Transcription
Whisper API transcribes the audio with precise timestamps.

### Step 3: Translation
Each segment is translated individually with GPT-5 mini, optimizing for lip-sync.

### Step 4: Voice Generation
Chatterbox TTS generates speech matching the voice prompt using voice cloning.

### Step 5: Video Assembly
FFmpeg merges the new audio with the original video.

## Troubleshooting

### FFmpeg not found

**Error**: `ffmpeg: command not found`

**Solution**: Install FFmpeg and add to PATH
- Windows: https://ffmpeg.org/download.html#build-windows
- Mac: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg`

### CUDA errors

**Error**: `CUDA out of memory` or `CUDA not available`

**Solution**: Use CPU instead:
```bash
python translate.py input.mp4 voice.wav output.mp4 --device cpu
```

### OpenAI API errors

**Error**: `OPENAI_API_KEY not found`

**Solution**: Create `.env` file with your API key or pass with `--api-key`

### Voice cloning sounds bad

**Solution**: Improve your voice prompt:
- Use high-quality, clean audio
- 3-10 seconds duration
- Single speaker, no background noise
- Consistent speaking style

## Current Limitations

This minimal implementation does NOT handle:

- Time-stretching to match original segment duration
- Silence preservation between segments
- Error retry logic
- Manual editing capabilities
- Queue/batch processing

## Next Steps

Potential improvements:

1. **Time-stretching**: Adjust audio duration to match original timestamps
2. **Silence handling**: Preserve natural pauses between segments
3. **CLI enhancements**: Progress bars, verbose mode
4. **Modular design**: Separate modules for each pipeline stage
5. **Wav2Lip integration**: Lip-sync enhancement

## Documentation Links

- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- [Whisper API Guide](https://platform.openai.com/docs/guides/speech-to-text)
- [GPT Responses API](https://platform.openai.com/docs/guides/responses)
- [Chatterbox GitHub](https://github.com/resemble-ai/chatterbox)

## License

MIT License - Feel free to modify and use as needed.
