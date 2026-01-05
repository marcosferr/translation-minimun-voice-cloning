#!/usr/bin/env python3
"""
Minimal Translation Pipeline with Voice Cloning

Pipeline:
    FFmpeg -> Whisper API -> gpt-5-mini (voice selection) -> gpt-5-mini (translation) -> Chatterbox TTS -> FFmpeg

Usage:
    python translate.py input.mp4 output.mp4 --source-lang es --target-lang en
    python translate.py input.mp4 output.mp4 --voice-prompt voice.wav --source-lang es --target-lang en
"""

import argparse
import os
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from typing import List

import numpy as np
try:
    import soundfile as sf
except ImportError:
    print(
        "Error: required dependency 'soundfile' (PySoundFile) is not installed.\n"
        "Install dependencies with:\n  pip install -r requirements.txt\n"
        "Or with conda (recommended on Windows):\n  conda install -c conda-forge pysoundfile libsndfile"
    )
    sys.exit(1)
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# Chatterbox import - optional dependency handling
try:
    from chatterbox.tts import ChatterboxTTS
except ImportError:
    print("Warning: chatterbox-tts not installed. Install with: pip install chatterbox-tts")
    sys.exit(1)

# Load environment variables
load_dotenv()


# Pydantic models for structured outputs
class VoiceSegmentSelection(BaseModel):
    """Model for voice segment selection response"""
    selected: List[int] = Field(description="List of selected segment indices for voice cloning")
    reason: str = Field(description="Brief explanation of why these segments were chosen")


class TranslatedSegment(BaseModel):
    """Model for a single translated segment"""
    original_text: str = Field(description="The original text in source language")
    translated_text: str = Field(description="The translated text in target language")
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")


class TranslationResponse(BaseModel):
    """Model for batch translation response"""
    segments: List[TranslatedSegment] = Field(description="List of all translated segments")


class TranslationPipeline:
    """Minimal translation pipeline with voice cloning"""

    def __init__(
        self,
        video_input: str,
        video_output: str,
        voice_prompt: str = None,
        source_lang: str = "es",
        target_lang: str = "en",
        device: str = "cuda",
        openai_api_key: str = None,
        voice_prompt_duration: float = 5.0,
    ):
        self.video_input = Path(video_input)
        self.voice_prompt = Path(voice_prompt) if voice_prompt else None
        self.video_output = Path(video_output)
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.device = device
        self.voice_prompt_duration = voice_prompt_duration

        # Initialize OpenAI client
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Set it as environment variable or pass it directly."
            )
        self.client = OpenAI(api_key=api_key)

        # Temporary files
        self.temp_dir = Path(tempfile.mkdtemp())
        self.audio_extracted = self.temp_dir / "audio.wav"
        self.auto_voice_prompt = self.temp_dir / "auto_voice_prompt.wav"
        self.tts_output = self.temp_dir / "tts.wav"

        # Locate ffmpeg executable
        # Hardcoded path from user's Windows installation
        hardcoded_ffmpeg = r"C:\Users\ferre\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"
        
        if os.path.exists(hardcoded_ffmpeg):
            self.ffmpeg_path = hardcoded_ffmpeg
            print(f"Using hardcoded FFmpeg path: {self.ffmpeg_path}")
        else:
            self.ffmpeg_path = shutil.which("ffmpeg")

        if not self.ffmpeg_path:
            raise FileNotFoundError(
                "FFmpeg not found. Please install FFmpeg and ensure it's available on PATH. "
                "On Windows, download a static build from https://ffmpeg.org/download.html or use a package manager like Chocolatey ('choco install ffmpeg')."
            )

    def extract_audio(self):
        """Extract audio from video using FFmpeg"""
        print(f"[1/6] Extracting audio from {self.video_input}...")

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i",
            str(self.video_input),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(self.audio_extracted),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")

        print(f"      Audio extracted to {self.audio_extracted}")

    def transcribe(self):
        """Transcribe audio using Whisper API with timestamps"""
        print(f"[2/6] Transcribing audio with Whisper API...")

        with open(self.audio_extracted, "rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="verbose_json",
                language=self.source_lang,
                timestamp_granularities=["segment"],
            )

        segments = response.segments
        print(f"      Transcription complete: {len(segments)} segments")

        return segments

    def select_best_voice_segments(self, segments):
        """Use gpt-5-mini to select the best voice segments for cloning"""
        print(f"[3/6] Selecting best voice segments with gpt-5-mini...")

        # Filter segments suitable for voice cloning (2-10 seconds, clear speech)
        suitable_segments = []
        for seg in segments:
            duration = seg.end - seg.start
            # Prefer segments between 2-8 seconds
            if 2.0 <= duration <= 8.0:
                suitable_segments.append({
                    "index": len(suitable_segments),
                    "text": seg.text,
                    "start": seg.start,
                    "end": seg.end,
                    "duration": duration,
                })

        if len(suitable_segments) < 3:
            print(f"      Warning: Only {len(suitable_segments)} suitable segments found")
            # Use all segments if not enough suitable ones
            suitable_segments = [
                {
                    "index": i,
                    "text": seg.text,
                    "start": seg.start,
                    "end": seg.end,
                    "duration": seg.end - seg.start,
                }
                for i, seg in enumerate(segments[:10])
            ]

        # Use gpt-5-mini with structured output to select best segments
        segments_info = "\n".join([
            f"Segment {s['index']}: {s['duration']:.1f}s - \"{s['text'][:50]}...\""
            for s in suitable_segments[:10]
        ])

        prompt = f"""Analyze these speech segments and select the 2-3 BEST segments for voice cloning.

Criteria for selection:
1. Clear speech without stammering or hesitation
2. Natural speaking pace (not too fast/slow)
3. Representative of the speaker's normal voice
4. No strong background noise or music
5. Total duration should be around {self.voice_prompt_duration} seconds
6. Prefer segments with complete sentences or phrases

Available segments:
{segments_info}

Use the index numbers from the segments list above."""

        response = self.client.responses.parse(
            model="gpt-5-mini",
            input=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing speech quality for voice cloning.",
                },
                {"role": "user", "content": prompt},
            ],
            text_format=VoiceSegmentSelection,
        )

        result = response.output_parsed
        selected_indices = result.selected if result.selected else list(range(min(3, len(suitable_segments))))
        reason = result.reason if result.reason else ""

        print(f"      Selected segments: {selected_indices}")
        print(f"      Reason: {reason}")

        # Get the actual segment objects
        selected_segments = [suitable_segments[i] for i in selected_indices if i < len(suitable_segments)]

        return selected_segments

    def extract_and_merge_voice_segments(self, segments):
        """Extract audio segments from original audio and merge them"""
        print(f"[4/6] Extracting and merging {len(segments)} voice segments...")

        # Load the full audio
        audio_data, sample_rate = sf.read(str(self.audio_extracted))

        audio_chunks = []
        total_duration = 0.0

        for seg in segments:
            start_sample = int(seg["start"] * sample_rate)
            end_sample = int(seg["end"] * sample_rate)

            # Extract segment
            segment_audio = audio_data[start_sample:end_sample]
            audio_chunks.append(segment_audio)

            duration = len(segment_audio) / sample_rate
            total_duration += duration
            print(f"      Segment {seg['index']}: {duration:.2f}s - \"{seg['text'][:40]}...\"")

        # Add small silence between segments (0.3s)
        silence_samples = int(0.3 * sample_rate)
        silence = np.zeros(silence_samples, dtype=audio_data.dtype)

        # Merge with silence between segments
        merged_audio = []
        for i, chunk in enumerate(audio_chunks):
            merged_audio.append(chunk)
            if i < len(audio_chunks) - 1:
                merged_audio.append(silence)

        final_audio = np.concatenate(merged_audio)
        final_duration = len(final_audio) / sample_rate

        sf.write(str(self.auto_voice_prompt), final_audio, sample_rate)
        print(f"      Voice prompt created: {final_duration:.2f}s duration")

        return str(self.auto_voice_prompt)

    def auto_select_voice_prompt(self, segments):
        """Automatically select and create voice prompt from video"""
        print(f"\n[Auto Voice Prompt] No voice prompt provided, extracting from video...")

        best_segments = self.select_best_voice_segments(segments)
        voice_prompt_path = self.extract_and_merge_voice_segments(best_segments)

        return voice_prompt_path

    def translate_segments(self, segments):
        """Translate all segments in one API call using gpt-5-mini with structured output"""
        print(f"[5/6] Translating {len(segments)} segments with gpt-5-mini (batch)...")

        # Build the segments info for the prompt
        segments_data = []
        for i, seg in enumerate(segments):
            segments_data.append({
                "index": i,
                "text": seg.text,
                "start": seg.start,
                "end": seg.end,
            })

        segments_json = "\n".join([
            f'Segment {s["index"]}: (start: {s["start"]:.2f}s, end: {s["end"]:.2f}s) "{s["text"]}"'
            for s in segments_data
        ])

        prompt = f"""Translate ALL the following segments from {self.source_lang} to {self.target_lang}.

For each segment:
- Maintain the original meaning
- Adjust length and rhythm to facilitate lip-sync synchronization
- Keep translations concise and natural-sounding

Segments to translate:
{segments_json}

Return ALL segments with their translations, preserving the exact start and end times."""

        response = self.client.responses.parse(
            model="gpt-5-mini",
            input=[
                {
                    "role": "system",
                    "content": f"You are an expert translator specializing in {self.source_lang} to {self.target_lang} translation for video dubbing. Translate all segments accurately while optimizing for lip-sync.",
                },
                {"role": "user", "content": prompt},
            ],
            text_format=TranslationResponse,
        )

        result = response.output_parsed
        translated_segments = []

        for seg in result.segments:
            translated_segments.append({
                "text": seg.translated_text,
                "start": seg.start,
                "end": seg.end,
            })
            print(f'      [{seg.start:.2f}s-{seg.end:.2f}s]: "{seg.translated_text[:40]}..."')

        print(f"      Translation complete: {len(translated_segments)} segments")
        return translated_segments

    def generate_speech(self, translated_segments, voice_prompt_path):
        """Generate speech using Chatterbox TTS with voice cloning"""
        print(f"[6/6] Generating speech with Chatterbox TTS ({self.device})...")

        # Optimize for 4GB VRAM
        tts_model = ChatterboxTTS.from_pretrained(device=self.device)
        audio_chunks = []

        for i, seg in enumerate(translated_segments, 1):
            print(f"      Segment {i}/{len(translated_segments)}: ", end="", flush=True)

            audio = tts_model.generate(
                text=seg["text"],
                audio_prompt_path=voice_prompt_path,
                temperature=0.6,
            )

            # audio is a tensor, not a dict
            audio_chunks.append(audio)
            print(f"Generated {len(audio)} samples")

        # Concatenate all audio chunks
        final_audio = sum(audio_chunks)
        sf.write(str(self.tts_output), final_audio, 24000)

        print(f"      TTS audio saved to {self.tts_output}")

        return final_audio

    def merge_audio_video(self):
        """Merge TTS audio with original video using FFmpeg"""
        print(f"\n[Merge] Merging audio and video...")

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i",
            str(self.video_input),
            "-i",
            str(self.tts_output),
            "-c:v",
            "copy",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-shortest",
            str(self.video_output),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")

        print(f"      Video saved to {self.video_output}")

    def run(self):
        """Execute the complete pipeline"""
        try:
            print(f"\n{'='*60}")
            print(f"Translation Pipeline: {self.source_lang} -> {self.target_lang}")
            print(f"Device: {self.device}")
            print(f"{'='*60}\n")

            # Extract audio
            self.extract_audio()

            # Transcribe
            segments = self.transcribe()

            # Determine voice prompt
            if self.voice_prompt and self.voice_prompt.exists():
                print(f"\n[Voice Prompt] Using provided: {self.voice_prompt}")
                voice_prompt_path = str(self.voice_prompt)
            else:
                voice_prompt_path = self.auto_select_voice_prompt(segments)

            # Translate
            translated_segments = self.translate_segments(segments)

            # Generate speech
            self.generate_speech(translated_segments, voice_prompt_path)

            # Merge
            self.merge_audio_video()

            print(f"\n{'='*60}")
            print(f"SUCCESS! Video generated: {self.video_output}")
            print(f"{'='*60}\n")

        finally:
            # Cleanup temporary files
            if self.temp_dir.exists():
                import shutil

                shutil.rmtree(self.temp_dir)
                print(f"Cleaned up temporary files")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Minimal translation pipeline with automatic voice cloning"
    )
    parser.add_argument("video_input", help="Input video file (e.g., input.mp4)")
    parser.add_argument("video_output", help="Output video file (e.g., output.mp4)")
    parser.add_argument(
        "--voice-prompt",
        help="Voice reference audio (optional, auto-selected if not provided)",
    )
    parser.add_argument(
        "--source-lang",
        default="es",
        help="Source language code (default: es)",
    )
    parser.add_argument(
        "--target-lang",
        default="en",
        help="Target language code (default: en)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for Chatterbox TTS (default: cuda)",
    )
    parser.add_argument(
        "--voice-duration",
        type=float,
        default=5.0,
        help="Target duration for auto voice prompt in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )

    args = parser.parse_args()

    # Validate input files
    if not Path(args.video_input).exists():
        print(f"Error: Video input not found: {args.video_input}")
        sys.exit(1)

    if args.voice_prompt and not Path(args.voice_prompt).exists():
        print(f"Error: Voice prompt not found: {args.voice_prompt}")
        sys.exit(1)

    # Run pipeline
    pipeline = TranslationPipeline(
        video_input=args.video_input,
        video_output=args.video_output,
        voice_prompt=args.voice_prompt,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        device=args.device,
        openai_api_key=args.api_key,
        voice_prompt_duration=args.voice_duration,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
