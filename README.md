# Sound2words
Transcripting experience with open-ai-whisper

## Kurulum
Bu projeyi çalıştırmak için Python ortamınızı hazırlamak üzere şu adımları izleyin:

1. **Sanal ortam oluşturun (opsiyonel ama önerilir):**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows için
   source .venv/bin/activate  # Linux/macOS için
   ```

2. **Bağımlılıkları yükleyin:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **CUDA kullanıyorsanız (opsiyonel):**
   NVIDIA GPU için uygun PyTorch sürümünü kurun. Örneğin CUDA 12.1 için:
   ```bash
   pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

> Not: `ffmpeg` sisteminizde ayrıca kurulu olmalı ve PATH’e eklenmelidir.

## Kullanım
```bash
python transkript.py "C:\...\audio.m4a"                # openai-whisper (varsayılan)
python transkript.py "C:\...\audio.m4a" --fw           # faster-whisper
python transkript.py --fw --model large-v3 --lang tr   # FW + large-v3 + Türkçe
python transkript.py                                   # Dosya seçme penceresi açılır
```

### Çıktılar
- `.txt` — düz metin
- `.srt` — altyazı dosyası (SRT formatı)
- `.vtt` — altyazı dosyası (WebVTT formatı)

