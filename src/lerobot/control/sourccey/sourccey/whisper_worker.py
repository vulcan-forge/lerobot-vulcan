#!/usr/bin/env python3
import argparse
import sys
import numpy as np
from faster_whisper import WhisperModel
import os

try:
    import soundfile as sf
except Exception:  # optional; only used for .wav/.flac/etc convenience
    sf = None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="medium.en")
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute-type", default="float16")
    p.add_argument("--language", default="en")
    p.add_argument("--beam-size", type=int, default=3)
    p.add_argument("--vad-filter", action="store_true")
    p.add_argument("--prompt", default="")
    p.add_argument("--audio", required=True)  # path to raw PCM16 or a common audio file (wav/flac/ogg)
    p.add_argument("--sample-rate", type=int, default=16000, help="Sample rate for raw PCM16 input.")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    path = args.audio
    ext = os.path.splitext(path)[1].lower()

    audio_f32: np.ndarray
    if ext in {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aiff", ".aif", ".aifc"}:
        if sf is None:
            print("[whisper_worker] soundfile not available; cannot read this audio format.", file=sys.stderr)
            return 2
        try:
            audio, sr = sf.read(path, dtype="float32", always_2d=False)
        except Exception as e:
            print(f"[whisper_worker] failed to read audio file: {e}", file=sys.stderr)
            return 2

        if audio is None:
            print("")
            return 0
        if audio.ndim > 1:
            # downmix
            audio = np.mean(audio, axis=-1)
        if int(sr) != 16000:
            print(f"[whisper_worker] expected 16kHz audio, got {sr}Hz (please resample).", file=sys.stderr)
            return 2
        audio_f32 = np.asarray(audio, dtype=np.float32)
    else:
        # raw PCM16
        audio_i16 = np.fromfile(path, dtype=np.int16)
        if audio_i16.size == 0:
            print("")
            return 0
        audio_f32 = audio_i16.astype(np.float32) / 32768.0

    if args.debug:
        rms = float(np.sqrt(np.mean(audio_f32.astype(np.float32) ** 2))) if audio_f32.size else 0.0
        dur_s = float(audio_f32.size) / 16000.0
        print(f"[whisper_worker] dur_s={dur_s:.2f} rms={rms:.6f}", file=sys.stderr)

    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
    )

    segments, _ = model.transcribe(
        audio_f32,
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=args.vad_filter,
        initial_prompt=args.prompt,
    )

    out = []
    for seg in segments:
        t = (seg.text or "").strip()
        if t:
            out.append(t)

    print(" ".join(out).strip())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
