# -*- coding: utf-8 -*-
# transkript.py — tek tıkla dosya seç + openai-whisper / faster-whisper seçilebilir
# Kullanım:
#   python transkript.py "C:\...\audio.m4a"                (openai-whisper)
#   python .\transkript.py --fw --model large-v3 --vad     (faster-whisper)
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

    if args.fw:
        # ------------------ faster-whisper yolu ------------------
        import subprocess, threading, sys
        from faster_whisper import WhisperModel

        def get_media_duration_sec(path: str) -> float:
            try:
                out = subprocess.check_output([
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    path
                ], stderr=subprocess.STDOUT)
                return float(out.decode().strip())
            except Exception:
                return 0.0

        class Spinner:
            def __init__(self, interval=0.5):
                self.interval = interval
                self._stop = threading.Event()
                self._thr = threading.Thread(target=self._run, daemon=True)
            def start(self):
                self._thr.start()
            def stop(self):
                self._stop.set()
                self._thr.join(timeout=1.0)
            def _run(self):
                ticks = 0
                while not self._stop.is_set():
                    ticks += 1
                    print(f"[FW] Başlatılıyor… (warmup {ticks*self.interval:.1f}s)", flush=True)
                    self._stop.wait(self.interval)

        # VRAM’e göre compute_type
        compute_type = "int8"
        if device == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            compute_type = "float16" if vram_gb >= 10 else "int8_float16"

        print(f"[FW] Model yükleniyor ({model_name}, compute_type={compute_type})…")
        t0 = time.time()
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        print(f"[FW] Model yüklendi. Süre: {time.time() - t0:.1f} sn")

        total_sec = get_media_duration_sec(str(audio_path))
        if total_sec > 0:
            print(f"[FW] Tahmini süre: {total_sec:.1f} sn")

        print("[FW] Transkripsiyon başladı…")
        t1 = time.time()

        # 1) Spinner'ı başlat (ilk segmente kadar nabız)
        spin = Spinner(interval=0.5)
        spin.start()
        got_first = False

        seg_iter, info = model.transcribe(
            str(audio_path),
            language=args.lang,         # None -> otomatik; --lang tr ise daha hızlı başlar
            task="transcribe",
            beam_size=1,                # hızlı başlangıç
            vad_filter=args.vad,        # --vad gecikme yaratabilir; gerek yoksa kapat
            # word_timestamps=False,
        )

        segs, texts = [], []
        last_print = time.time()
        processed = 0.0

        for s in seg_iter:
            if not got_first:
                # 2) İlk segment geldi: spinner'ı sustur
                spin.stop()
                got_first = True
                print("[FW] İlk segment alındı, ilerleme başlıyor…", flush=True)

            segs.append({"start": s.start, "end": s.end, "text": s.text})
            texts.append(s.text)

            processed = max(processed, float(getattr(s, "end", 0.0)))
            now = time.time()
            if now - last_print >= 0.5:
                if total_sec > 0:
                    pct = min(100.0, processed / total_sec * 100.0)
                    print(f"[FW] {processed:7.1f}s / {total_sec:7.1f}s  ({pct:5.1f}%)", flush=True)
                else:
                    print(f"[FW] İşlenen ~{processed:.1f}s", flush=True)
                last_print = now

        # Eğer hiç segment gelmediyse spinner çalışıyor olabilir; kapat
        if not got_first:
            spin.stop()

        dur = time.time() - t1
        print(f"[FW] Transkripsiyon tamamlandı! Süre: {dur:.1f} sn", flush=True)

        result_dict = {
            "text": " ".join(t.strip() for t in texts).strip(),
            "segments": segs,
            "language": getattr(info, "language", None),
        }

        # --- TEŞHİS: nereye yazacağız? bu script hangisi? çalışma dizini ne? ---
        print(f"[DBG] __file__        : {__file__}", flush=True)
        print(f"[DBG] cwd            : {os.getcwd()}", flush=True)
        print(f"[DBG] audio_path     : {audio_path}", flush=True)
        print(f"[DBG] outdir (target): {outdir}", flush=True)
        print(f"[DBG] out_txt        : {out_txt}", flush=True)

        # --- TEŞHİS: bu klasöre yazabiliyor muyuz? ---
        try:
            testfile = outdir / "_write_test.tmp"
            testfile.write_text("ok", encoding="utf-8")
            print(f"[DBG] Write test OK: {testfile}", flush=True)
            testfile.unlink(missing_ok=True)
        except Exception as e:
            print(f"[ERR] Klasöre yazılamıyor: {outdir} | Hata: {e}", flush=True)

        # --- ÇIKTI DOSYALARINI YAZ ---
        try:
            dump_txt(result_dict, out_txt)
            dump_srt(result_dict, out_srt)
            dump_vtt(result_dict, out_vtt)
            print(f"[OUT] Kaydedildi:\n  TXT: {out_txt}\n  SRT: {out_srt}\n  VTT: {out_vtt}", flush=True)

            # Yazıldı mı gerçekten? (Windows/OneDrive için doğrula)
            for p in (out_txt, out_srt, out_vtt):
                print(f"[CHK] {p}  exists={p.exists()}  size={p.stat().st_size if p.exists() else 0}", flush=True)
        except Exception as e:
            print(f"[ERR] Çıktı yazılamadı: {e}", flush=True)

if __name__ == "__main__":
    main()
