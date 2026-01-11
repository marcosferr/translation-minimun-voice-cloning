#!/usr/bin/env python3
"""
Script de Clonación de Voz Simple usando Chatterbox

Este script extrae el audio de un video (mp4) y lo utiliza como referencia (prompt)
para clonar la voz y generar un texto predefinido usando Chatterbox TTS.

Uso:
    python clone_voice.py input.mp4 output.wav --lang es
"""

import argparse
import os
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

import torch
import soundfile as sf
import numpy as np

try:
    from chatterbox.tts_turbo import ChatterboxTurboTTS
except ImportError:
    print("Advertencia: No se pudo importar ChatterboxTurboTTS. Intentando importar ChatterboxTTS estándar...")
    try:
        from chatterbox.tts import ChatterboxTTS as ChatterboxTurboTTS
    except ImportError:
        print("Error: chatterbox-tts no está instalado. Instálalo con: pip install chatterbox-tts")
        sys.exit(1)

# Textos constantes para generar según el idioma seleccionado
# Incluyen tags paralingüísticos para probar la expresividad con Chatterbox
TEXTS = {
    "es": "Hola, esta es una prueba de clonación de voz automática con mucha expresión. [laugh] ¡Es increíble cómo suena! [chuckle] Aunque a veces es un poco extraño. [sigh] Qué alivio que funcione. [gasp] ¡Oh, qué sorpresa! [cough] Perdón, tengo tos. [sniff] ¿Hueles eso? [groan] Ay, qué cansancio. [shush] Shhh, escucha esto. [clear throat] Bueno, continuemos con la prueba.",
    "en": "Hello, this is an automatic voice cloning test with high expression. [laugh] It's amazing how it sounds! [chuckle] Though sometimes it's a bit strange. [sigh] What a relief it works. [gasp] Oh, what a surprise! [cough] Sorry, I have a cough. [sniff] Do you smell that? [groan] Ugh, so tired. [shush] Shhh, listen to this. [clear throat] Anyway, let's continue with the test.",
    "fr": "Bonjour, ceci est un test de clonage de voix automatique. [laugh] C'est incroyable! [sigh] Quel soulagement. [gasp] Oh, quelle surprise!",
    "pt": "Olá, este é um teste de clonagem de voz automática. [laugh] É incrível! [sigh] Que alívio."
}

class VoiceCloner:
    def __init__(self, video_input, output_audio, language="es", device="cuda"):
        self.video_input = Path(video_input)
        self.output_audio = Path(output_audio)
        self.language = language
        self.device = device
        
        # Seleccionar texto basado en el idioma
        self.text_to_generate = TEXTS.get(language, TEXTS["es"])
        
        # Directorio temporal
        self.temp_dir = Path(tempfile.mkdtemp())
        self.audio_extracted = self.temp_dir / "voice_prompt.wav"

        # Localizar ffmpeg (usando la misma lógica que en translate.py)
        hardcoded_ffmpeg = r"C:\Users\ferre\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"
        
        if os.path.exists(hardcoded_ffmpeg):
            self.ffmpeg_path = hardcoded_ffmpeg
        else:
            self.ffmpeg_path = shutil.which("ffmpeg")

        if not self.ffmpeg_path:
            raise FileNotFoundError(
                "FFmpeg no encontrado. Por favor instala FFmpeg y asegúrate de que esté en el PATH."
            )

    def extract_audio(self):
        """Extrae el audio del video usando FFmpeg"""
        print(f"[1/3] Extrayendo audio de {self.video_input}...")

        # Extraer audio a 16kHz mono (formato ideal para muchos modelos TTS)
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
            raise RuntimeError(f"FFmpeg falló: {result.stderr}")

        print(f"      Audio extraído a {self.audio_extracted}")

    def generate_speech(self):
        """Genera el audio clonado usando ChatterboxTurboTTS"""
        print(f"[2/3] Cargando modelo ChatterboxTurboTTS en {self.device}...")
        
        try:
            tts_model = ChatterboxTurboTTS.from_pretrained(device=self.device)
        except Exception as e:
            print(f"Error cargando el modelo: {e}")
            sys.exit(1)

        print(f"[3/3] Generando voz clonada con expresividad...")
        print(f"      Texto: \"{self.text_to_generate}\"")
        print(f"      Prompt de voz: {self.audio_extracted}")

        # Generar audio con el modelo Turbo
        try:
            audio = tts_model.generate(
                text=self.text_to_generate,
                audio_prompt_path=str(self.audio_extracted),
                temperature=0.7,          # Variabilidad
                exaggeration=0.8,         # Expresividad (soportado en Turbo)
                cfg_weight=0.4,           # Adherencia al prompt
            )
        except Exception as e:
            print(f"Advertencia: Error con parámetros avanzados ({e}). Reintentando sin parámetros extra...")
            audio = tts_model.generate(
                text=self.text_to_generate,
                audio_prompt_path=str(self.audio_extracted),
                temperature=0.7,
            )

        # Procesar salida
        if isinstance(audio, torch.Tensor):
            final_audio = audio.cpu().float().numpy().flatten()
        else:
            final_audio = audio

        # Guardar resultado (24kHz es común para salidas de TTS modernos, ajustar si es necesario)
        sf.write(str(self.output_audio), final_audio, 24000)
        print(f"\n¡Éxito! Audio generado guardado en: {self.output_audio}")

    def run(self):
        try:
            if not self.video_input.exists():
                print(f"Error: El archivo de entrada {self.video_input} no existe.")
                return

            self.extract_audio()
            self.generate_speech()
            
        except Exception as e:
            print(f"Ocurrió un error inesperado: {e}")
        finally:
            # Limpieza
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                print("      Archivos temporales eliminados.")

def main():
    parser = argparse.ArgumentParser(description="Clonación de voz simple desde MP4 usando Chatterbox")
    parser.add_argument("video_input", help="Archivo de video de entrada (mp4)")
    parser.add_argument("output_audio", help="Archivo de audio de salida (wav)")
    parser.add_argument("--lang", default="es", choices=["es", "en", "fr", "pt"], help="Idioma del texto a generar (default: es)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Dispositivo a usar (default: cuda)")

    args = parser.parse_args()

    cloner = VoiceCloner(
        video_input=args.video_input,
        output_audio=args.output_audio,
        language=args.lang,
        device=args.device
    )
    
    cloner.run()

if __name__ == "__main__":
    main()
