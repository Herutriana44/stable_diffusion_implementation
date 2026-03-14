# Stable Diffusion Inpainting - Flask App

Aplikasi Flask untuk inference Stable Diffusion inpainting dengan model dari Civit AI.

## Fitur

- **Input wajib:** Gambar, prompt, mask inpainting (digambar di web)
- **Model:** Civit AI (URL + API key)
- **Format:** Checkpoint safetensor (.safetensors) atau .ckpt
- **Inpainting di web:** Upload gambar → gambar mask di area yang ingin diubah → isi prompt → inference

## Cara Menjalankan

```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan server (GPU disarankan)
python app.py
```

Buka http://localhost:5000 di browser.

## Alur Kerja

1. **Load Model:** Masukkan URL Civit AI dan API key (opsional), klik "Load Model"
2. **Upload Gambar:** Pilih gambar yang ingin di-edit
3. **Gambar Mask:** Gambar di area yang ingin di-inpaint (putih = area yang akan diubah)
4. **Prompt:** Isi deskripsi untuk area yang di-mask
5. **Inference:** Klik "Jalankan Inpainting"

## Model

Gunakan model **inpainting** dari Civit AI. Cari "stable diffusion inpainting" atau "sd 1.5 inpainting" di Civit AI.

Contoh: https://civitai.com/models/9427/stable-diffusion-15-inpainting

## Persyaratan

- Python 3.10+
- GPU dengan CUDA (disarankan) atau CPU
- ~4GB+ VRAM untuk inference
