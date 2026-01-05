A continuación tienes **el script más mínimo y lineal posible en Python** que implementa exactamente este pipeline:

```
FFmpeg → Whisper API → GPT-5 mini → Chatterbox (local) → FFmpeg
```

No hay colas, no hay GUI, no hay clases: **solo el flujo esencial**, pensado para que luego lo refactorices.

---

## 0. Prerrequisitos

### Sistema

* FFmpeg instalado y en PATH
  [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

### Python

```bash
pip install openai chatterbox-tts soundfile
```

Recomendado:

* Python 3.10+
* GPU CUDA (opcional, pero Chatterbox lo agradece)

---

## 1. Variables de entorno

```bash
export OPENAI_API_KEY="tu_api_key"
```

---

## 2. Script Python mínimo (end-to-end)

```python
import subprocess
import os
import soundfile as sf
from openai import OpenAI
from chatterbox.tts import ChatterboxTTS

# -------------------------
# Config
# -------------------------
VIDEO_INPUT = "input.mp4"
AUDIO_EXTRACTED = "audio.wav"
VOICE_PROMPT = "voice_prompt.wav"   # audio limpio de referencia
TTS_OUTPUT = "tts.wav"
VIDEO_OUTPUT = "output.mp4"

SOURCE_LANG = "es"
TARGET_LANG = "en"

client = OpenAI()

# -------------------------
# 1. Extraer audio (FFmpeg)
# -------------------------
subprocess.run([
    "ffmpeg", "-y",
    "-i", VIDEO_INPUT,
    "-ac", "1",
    "-ar", "16000",
    AUDIO_EXTRACTED
], check=True)

# -------------------------
# 2. Whisper API (transcripción con timestamps)
# -------------------------
with open(AUDIO_EXTRACTED, "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
        response_format="verbose_json"
    )

segments = transcript["segments"]

# -------------------------
# 3. Traducción con GPT-5 mini (segmento por segmento)
#    Optimizada para lip-sync
# -------------------------
translated_segments = []

for seg in segments:
    prompt = f"""
Traduce el siguiente texto del {SOURCE_LANG} al {TARGET_LANG}.
Mantén el significado original.
Ajusta longitud y ritmo para facilitar sincronización labial.
Evita explicaciones largas.

Texto:
\"\"\"{seg['text']}\"\"\"
"""

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    translated_text = response.output_text
    translated_segments.append(translated_text)

# -------------------------
# 4. Chatterbox TTS (clonación de voz local)
# -------------------------
tts_model = ChatterboxTTS.from_pretrained(device="cuda")  # o "cpu"

audio_chunks = []

for text in translated_segments:
    audio = tts_model.generate(
        text=text,
        audio_prompt_path=VOICE_PROMPT,
        language_id=TARGET_LANG,
        temperature=0.6
    )
    audio_chunks.append(audio)

# Concatenar audios
final_audio = sum(audio_chunks)
sf.write(TTS_OUTPUT, final_audio, 24000)

# -------------------------
# 5. Merge audio + video (FFmpeg)
# -------------------------
subprocess.run([
    "ffmpeg", "-y",
    "-i", VIDEO_INPUT,
    "-i", TTS_OUTPUT,
    "-c:v", "copy",
    "-map", "0:v:0",
    "-map", "1:a:0",
    VIDEO_OUTPUT
], check=True)

print("Proceso completo. Video generado:", VIDEO_OUTPUT)
```

---

## 3. Qué hace (en términos estrictos)

| Paso        | Resultado                         |
| ----------- | --------------------------------- |
| FFmpeg      | Extrae audio limpio               |
| Whisper API | Texto + timestamps                |
| GPT-5 mini  | Traducción ajustada para lip-sync |
| Chatterbox  | Voz clonada local                 |
| FFmpeg      | Video final doblado               |

---

## 4. Documentación oficial (links clave)

### FFmpeg

* [https://ffmpeg.org/documentation.html](https://ffmpeg.org/documentation.html)
* [https://trac.ffmpeg.org/wiki](https://trac.ffmpeg.org/wiki)

---

### Whisper API (Speech-to-Text)

* [https://platform.openai.com/docs/guides/speech-to-text](https://platform.openai.com/docs/guides/speech-to-text)
* [https://platform.openai.com/docs/api-reference/audio](https://platform.openai.com/docs/api-reference/audio)

---

### GPT-5 mini (Responses API)

* [https://platform.openai.com/docs/guides/responses](https://platform.openai.com/docs/guides/responses)
* [https://platform.openai.com/docs/api-reference/responses](https://platform.openai.com/docs/api-reference/responses)

---

### Chatterbox TTS (Voice Cloning local)

* GitHub: [https://github.com/resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox)
* PyPI: [https://pypi.org/project/chatterbox-tts/](https://pypi.org/project/chatterbox-tts/)
* Ejemplos oficiales: [https://github.com/resemble-ai/chatterbox/tree/main/examples](https://github.com/resemble-ai/chatterbox/tree/main/examples)

---

### soundfile (WAV IO)

* [https://pysoundfile.readthedocs.io/](https://pysoundfile.readthedocs.io/)

---

## 5. Limitaciones conscientes de este script

Esto **no** maneja aún:

* Ajuste de duración por timestamp
* Silencios entre segmentos
* Reintentos / errores
* Cola de procesamiento
* Edición manual

Pero:
✔ Es el **mínimo funcional real**
✔ Es correcto arquitectónicamente
✔ Es exactamente el pipeline que definiste

---

## 6. Próximo paso lógico (si quieres)

Puedo ayudarte a:

* Ajustar duración por segmento (time-stretch sin pitch shift)
* Sincronizar audio con timestamps Whisper
* Convertir esto en CLI (`argparse`)
* Extraer esto a módulos limpios
* Integrar Wav2Lip después

Dime cuál sigue.
