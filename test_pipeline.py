#!/usr/bin/env python3
"""
Test script for the translation pipeline
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and display output"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, text=True)

    if result.returncode != 0:
        print(f"\n❌ FAILED: {description}")
        return False
    else:
        print(f"\n✅ SUCCESS: {description}")
        return True

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("TRANSLATION PIPELINE - TEST SUITE")
    print("="*60)

    # Check if input.mp4 exists
    input_video = Path("input.mp4")
    if not input_video.exists():
        print(f"\n❌ Error: input.mp4 not found in current directory")
        print(f"Please place input.mp4 in: {Path.cwd()}")
        sys.exit(1)

    print(f"\n✅ Found input video: {input_video}")

    # Test 1: Check Python syntax
    print("\n" + "="*60)
    print("Test 1: Checking Python syntax")
    print("="*60)
    if not run_command([sys.executable, "-m", "py_compile", "translate.py"], "Syntax check"):
        sys.exit(1)

    # Test 2: Check imports
    print("\n" + "="*60)
    print("Test 2: Checking imports")
    print("="*60)
    test_imports = """
import sys
sys.path.insert(0, '.')

try:
    import numpy
    print('✅ numpy imported')
except ImportError as e:
    print(f'❌ numpy: {e}')
    sys.exit(1)

try:
    import soundfile
    print('✅ soundfile imported')
except ImportError as e:
    print(f'❌ soundfile: {e}')
    sys.exit(1)

try:
    from openai import OpenAI
    print('✅ openai imported')
except ImportError as e:
    print(f'❌ openai: {e}')
    sys.exit(1)

try:
    from chatterbox.tts import ChatterboxTTS
    print('✅ chatterbox imported')
except ImportError as e:
    print(f'❌ chatterbox: {e}')
    sys.exit(1)

print('\\n✅ All imports successful')
"""
    result = subprocess.run([sys.executable, "-c", test_imports], text=True)
    if result.returncode != 0:
        print("\n❌ Import test failed")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)

    # Test 3: Check FFmpeg
    print("\n" + "="*60)
    print("Test 3: Checking FFmpeg")
    print("="*60)
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ FFmpeg not found")
        sys.exit(1)
    print("✅ FFmpeg installed")

    # Test 4: Check OpenAI API key
    print("\n" + "="*60)
    print("Test 4: Checking OpenAI API key")
    print("="*60)
    check_env = """
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
if api_key and api_key != 'your_api_key_here':
    print(f'✅ OPENAI_API_KEY is set (starts with: {api_key[:7]}...)')
else:
    print('❌ OPENAI_API_KEY not set or invalid')
    print('Create a .env file with: OPENAI_API_KEY=sk-...')
    exit(1)
"""
    result = subprocess.run([sys.executable, "-c", check_env], text=True)
    if result.returncode != 0:
        sys.exit(1)

    # Test 5: Get video info
    print("\n" + "="*60)
    print("Test 5: Getting video information")
    print("="*60)
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(input_video)],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"✅ Video info retrieved")
        print(f"   File: {input_video}")
        print(f"   Size: {input_video.stat().st_size / (1024*1024):.2f} MB")
    else:
        print(f"❌ Could not read video info")
        sys.exit(1)

    # Test 6: Run translation (English -> Spanish)
    print("\n" + "="*60)
    print("Test 6: Running translation pipeline")
    print("Source: English (en)")
    print("Target: Spanish (es)")
    print("Auto voice prompt: Enabled")
    print("="*60)

    output_video = "output_test.mp4"
    cmd = [
        sys.executable, "translate.py",
        str(input_video),
        output_video,
        "--source-lang", "en",
        "--target-lang", "es",
        "--device", "cuda"
    ]

    print(f"\nCommand: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, text=True)

    if result.returncode == 0 and Path(output_video).exists():
        print(f"\n✅ Translation successful!")
        print(f"   Output: {output_video}")
        print(f"   Size: {Path(output_video).stat().st_size / (1024*1024):.2f} MB")
    else:
        print(f"\n❌ Translation failed")
        sys.exit(1)

    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✅")
    print("="*60)

if __name__ == "__main__":
    main()
