# -*- coding: utf-8 -*-
# transkript.py — tek tıkla dosya seç + openai-whisper / faster-whisper seçilebilir
# Kullanım:
#   python transkript.py "C:\...\audio.m4a"                (openai-whisper)
#   python transkript.py "C:\...\audio.m4a" --fw           (faster-whisper)
#   python transkript.py --fw --model large-v3 --lang tr   (FW + large-v3 + TR)
#   python transkript.py                                    (pencere açar)

import os
import sys
import time
import shutil
import argparse
from datetime import timedelta
from pathlib import Path

import torch

# openai-whisper (varsayılan backend)
import whisper as ow

# ---------- Yardımcılar ----------

def srt_timestamp(seconds: float) -> str:
    if seconds is None:
        seconds = 0.0
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def vtt_timestamp(seconds: float) -> str:
    if seconds is None:
        seconds = 0.0
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def dump_txt(result_dict, path_txt: Path):
    text = (result_dict.get("text") or "").strip()
    path_txt.write_text(text, encoding="utf-8")

def dump_srt(result_dict, path_srt: Path):
    segs = result_dict.get("segments", []) or []
    lines = []
    for i, seg in enumerate(segs, 1):
        start = srt_timestamp(seg.get("start"))
        end   = srt_timestamp(seg.get("end"))
        text  = (seg.get("text") or "").strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    path_srt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

def dump_vtt(result_dict, path_vtt: Path):
    segs = result_dict.get("segments", []) or []
    lines = ["WEBVTT", ""]
    for seg in segs:
        start = vtt_timestamp(seg.get("start"))
        end   = vtt_timestamp(seg.get("end"))
        text  = (seg.get("text") or "").strip()
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    path_vtt.write_text("\n".join(lines), encoding="utf-8")

def try_clipboard_path():
    """Panodan dosya yolu almayı dener (opsiyonel). Pyperclip yoksa sessiz geçer."""
    try:
        import pyperclip
        clip = pyperclip.paste().strip().strip('"')
        p = Path(clip)
        if clip and p.is_file():
            return p
    except Exception:
        pass
    return None

def file_dialog_pick(start_dir: Path | None = None) -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(
            initialdir=str(start_dir or Path.home()),
            title="Ses dosyasını seç (m4a/mp3/wav/flac/ogg/opus)",
            filetypes=[
                ("Audio files", "*.m4a *.mp3 *.wav *.flac *.ogg *.opus"),
                ("All files", "*.*"),
            ],
        )
        return Path(path) if path else None
    except Exception:
        return None

def resolve_audio_path(cli_path: str | None) -> Path:
    if cli_path:
        p = Path(cli_path.strip('"'))
        if p.is_file():
            return p
    p = try_clipboard_path()
    if p:
        return p
    p = file_dialog_pick()
    if p and p.is_file():
        return p
    while True:
        raw = input("Ses dosyası yolunu gir (sürükle-bırak + Enter): ").strip().strip('"')
        p = Path(raw)
        if p.is_file():
            return p
        print("Dosya bulunamadı, tekrar dene.")

def pick_device():
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    device_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    fp16_flag = bool(use_cuda)
    return device, device_name, fp16_flag

def auto_model_for_vram(prefer: str | None = None) -> str:
    """VRAM’a göre model seç: 12+GB -> large-v3, 6+GB -> medium, aksi -> small; CPU -> base."""
    if prefer:
        return prefer
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gb >= 6:
            return "large-v3"
        elif vram_gb >= 2:
            return "medium"
        else:
            return "small"
    return "base"

def build_uniform_result_text_segments_from_fw(segments_iter, info):
    """faster-whisper çıktılarını openai-whisper ile aynı sözlük yapısına dönüştürür."""
    segs = []
    texts = []
    for s in segments_iter:
        segs.append({"start": s.start, "end": s.end, "text": s.text})
        texts.append(s.text)
    return {
        "text": " ".join(t.strip() for t in texts).strip(),
        "segments": segs,
        "language": info.language if hasattr(info, "language") else None,
    }

# ---------- Ana akış ----------

def main():
    parser = argparse.ArgumentParser(
        description="Whisper ile kolay transkripsiyon (GUI/clipboard destekli)."
    )
    parser.add_argument("audio", nargs="?", help="Ses dosyası yolu (yoksa pencere açılır).")
    parser.add_argument("--model", default=None,
                        help='Model: tiny/base/small/medium/large-v3 (varsayılan: VRAM’a göre otomatik)')
    parser.add_argument("--lang", default=None,
                        help='Dil kodu, ör. "tr" (boş: otomatik).')
    parser.add_argument("--outdir", default=None,
                        help="Çıktı klasörü (boş: ses dosyasının klasörü).")
    parser.add_argument("--no-words", action="store_true",
                        help="Kelime zaman damgasını kapat (openai-whisper için).")
    parser.add_argument("--fw", action="store_true",
                        help="faster-whisper backend kullan (VRAM dostu, large-v3 bile çalışır).")
    parser.add_argument("--vad", action="store_true",
                        help="faster-whisper VAD filtresini aç (gürültüde faydalı).")

    args = parser.parse_args()

    audio_path = resolve_audio_path(args.audio).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else audio_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    base = audio_path.stem
    out_txt = outdir / f"{base}.txt"
    out_srt = outdir / f"{base}.srt"
    out_vtt = outdir / f"{base}.vtt"

    print("ffmpeg path (python içinden):", shutil.which("ffmpeg"))

    device, device_name, fp16_flag = pick_device()
    print(f"CUDA: {torch.cuda.is_available()} | Device: {device_name}")

    model_name = auto_model_for_vram(args.model)

    if not args.fw:
        # ------------------ openai-whisper yolu ------------------
        print(f"[OW] Model yükleniyor ({model_name})...")
        t0 = time.time()
        model = ow.load_model(model_name, device=device)
        print(f"[OW] Model yüklendi. Süre: {time.time() - t0:.1f} sn")

        initial_prompt = (
            "Türkçe ve İngilizce karışık bir konuşmanın düzgün noktalama ile yazıya dökümü. "
            "İmla ve özel isimler korunur. Gerekmedikçe kelime uydurma."
        )

        print("[OW] Transkripsiyon başladı…")
        t1 = time.time()
        result = model.transcribe(
            str(audio_path),
            task="transcribe",
            language=args.lang,
            fp16=fp16_flag,
            temperature=[0.0, 0.2, 0.4],
            beam_size=5,
            best_of=None,
            patience=0.0,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
            suppress_tokens="-1",
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            word_timestamps=not args.no_words,
            verbose=False,
        )
        dur = time.time() - t1
        print(f"[OW] Transkripsiyon tamamlandı! Süre: {dur:.1f} sn")

        # Zaten openai-whisper sözlüğü ile uyumlu
        result_dict = {
            "text": result.get("text", ""),
            "segments": result.get("segments", []),
            "language": result.get("language"),
        }

    else:
        # ------------------ faster-whisper yolu ------------------
        from faster_whisper import WhisperModel

        # VRAM’e göre compute_type seçimi:
        #  - 10GB+ -> float16
        #  - 6–10GB -> int8_float16  (RTX 4050 6GB için en iyi denge)
        #  - CPU -> int8
        compute_type = "int8"
        if device == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            compute_type = "float16" if vram_gb >= 10 else "int8_float16"

        print(f"[FW] Model yükleniyor ({model_name}, compute_type={compute_type})…")
        t0 = time.time()
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        print(f"[FW] Model yüklendi. Süre: {time.time() - t0:.1f} sn")

        print("[FW] Transkripsiyon başladı…")
        t1 = time.time()
        segments, info = model.transcribe(
            str(audio_path),
            language=args.lang,          # None -> otomatik
            task="transcribe",
            beam_size=5,
            vad_filter=args.vad,         # --vad ile aç
            # vad_parameters={"min_silence_duration_ms": 500}  # istersen ince ayar
        )
        result_dict = build_uniform_result_text_segments_from_fw(segments, info)
        dur = time.time() - t1
        print(f"[FW] Transkripsiyon tamamlandı! Süre: {dur:.1f} sn")

    # Çıktıları yaz
    dump_txt(result_dict, out_txt)
    dump_srt(result_dict, out_srt)
    dump_vtt(result_dict, out_vtt)

    print("Algılanan ana dil:", result_dict.get("language"))
    print("Toplam segment:", len(result_dict.get("segments", [])))
    print("TXT:", out_txt)
    print("SRT:", out_srt)
    print("VTT:", out_vtt)

if __name__ == "__main__":
    main()
