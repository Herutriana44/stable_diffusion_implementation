"""
Flask app untuk Stable Diffusion Inpainting.
Input: gambar (wajib), prompt, mask inpainting (digambar di web).
Model dari Civit AI (URL + API key).
Default URL dan API key bisa di-set via .env (CIVITAI_MODEL_URL, CIVITAI_API_KEY).
"""

import os
import re
import time
import uuid
import zipfile
from dotenv import load_dotenv

load_dotenv()
import io
import base64
import hashlib
import requests
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import torch
import prompt_templates

# Template prompt - format list, bisa diubah sesuai kebutuhan
PROMPT_TEMPLATES = prompt_templates.PROMPT_TEMPLATES

# Lazy load diffusers - hanya saat inference
_pipeline = None
_model_path = None

# Job progress untuk inpainting (background job + polling)
_inpaint_jobs = {}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload
app.config["MODEL_CACHE_DIR"] = Path(__file__).parent / "models"
app.config["UPLOAD_FOLDER"] = Path(__file__).parent / "uploads"
app.config["OUTPUT_FOLDER"] = Path(__file__).parent / "outputs"

# Buat folder
for folder in [app.config["MODEL_CACHE_DIR"], app.config["UPLOAD_FOLDER"], app.config["OUTPUT_FOLDER"]]:
    folder.mkdir(parents=True, exist_ok=True)


def get_civitai_download_url(url: str, api_key: str) -> str:
    """Dapatkan URL download dari Civit AI (direct URL atau model page)."""
    if "/api/download/models/" in url:
        return url.strip()

    match = re.search(r"civitai\.com/models/(\d+)", url)
    if not match:
        raise ValueError(
            "URL tidak valid. Gunakan: https://civitai.com/models/MODEL_ID atau "
            "https://civitai.com/api/download/models/VERSION_ID"
        )

    model_id = match.group(1)
    api_url = f"https://civitai.com/api/v1/models/{model_id}"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = requests.get(api_url, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    model_versions = data.get("modelVersions", [])
    if not model_versions:
        raise ValueError("Model tidak memiliki versi yang tersedia")

    for version in model_versions:
        files = version.get("files", [])
        primary_safetensor = next(
            (f for f in files if f.get("primary") and f.get("metadata", {}).get("format") == "SafeTensor"),
            None,
        )
        if primary_safetensor:
            return primary_safetensor.get("downloadUrl", "")
        safetensor = next((f for f in files if f.get("metadata", {}).get("format") == "SafeTensor"), None)
        if safetensor:
            return safetensor.get("downloadUrl", "")
        primary = next((f for f in files if f.get("primary")), None)
        if primary:
            return primary.get("downloadUrl", "")
        if files:
            return files[0].get("downloadUrl", "")

    first_version = model_versions[0]
    return f"https://civitai.com/api/download/models/{first_version['id']}"


def download_civitai_model(url: str, api_key: str, save_dir: Path) -> str:
    """Download model dari Civit AI."""
    download_url = get_civitai_download_url(url, api_key)
    save_dir.mkdir(parents=True, exist_ok=True)

    if api_key:
        sep = "&" if "?" in download_url else "?"
        download_url = f"{download_url}{sep}token={api_key}"

    # Cache by URL agar model sama tidak di-download ulang
    cache_key = hashlib.md5(url.strip().encode()).hexdigest()
    for ext in [".safetensors", ".ckpt"]:
        cached = save_dir / f"{cache_key}{ext}"
        if cached.exists():
            return str(cached)

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = requests.get(download_url, headers=headers, stream=True, timeout=120)
    resp.raise_for_status()

    content_disp = resp.headers.get("Content-Disposition")
    if content_disp and "filename=" in content_disp:
        filename = re.findall(r'filename[*]?=(?:UTF-8\'\')?["\']?([^"\';\n]+)', content_disp)
        filename = filename[0].strip() if filename else "model.safetensors"
    else:
        filename = "model.safetensors"

    ext = ".safetensors" if ".safetensors" in filename.lower() else ".ckpt"
    save_path = save_dir / f"{cache_key}{ext}"
    with open(save_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    return str(save_path)


def load_pipeline(model_path: str):
    """Load inpainting pipeline dari single file."""
    global _pipeline, _model_path
    if _pipeline is not None and _model_path == model_path:
        return _pipeline

    from diffusers import StableDiffusionInpaintPipeline

    _pipeline = StableDiffusionInpaintPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=model_path.endswith(".safetensors"),
    )
    _pipeline = _pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
    _model_path = model_path
    return _pipeline


@app.route("/")
def index():
    return render_template(
        "index.html",
        civitai_model_url=os.getenv("CIVITAI_MODEL_URL", ""),
        civitai_api_key=os.getenv("CIVITAI_API_KEY", ""),
        prompt_templates=PROMPT_TEMPLATES,
    )


@app.route("/api/load-model", methods=["POST"])
def load_model():
    """Load model dari Civit AI URL + API key."""
    data = request.get_json() or {}
    url = data.get("url", "").strip()
    api_key = data.get("api_key", "").strip()

    if not url:
        return jsonify({"success": False, "error": "URL model wajib diisi"}), 400

    try:
        model_path = download_civitai_model(url, api_key, app.config["MODEL_CACHE_DIR"])
        load_pipeline(model_path)
        return jsonify({"success": True, "message": "Model berhasil dimuat"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def _run_inpaint_job(job_id, image_b64, mask_b64, prompt, negative_prompt, strength, guidance_scale, num_steps, seed):
    """Jalankan inpainting di background thread."""
    global _inpaint_jobs
    start_time = time.time()
    _inpaint_jobs[job_id] = {"status": "running", "step": 0, "total_steps": num_steps, "elapsed_sec": 0}

    try:
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        image_data = base64.b64decode(image_b64)
        init_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        if "," in mask_b64:
            mask_b64 = mask_b64.split(",", 1)[1]
        mask_data = base64.b64decode(mask_b64)
        mask_image = Image.open(io.BytesIO(mask_data)).convert("L")

        if init_image.size != mask_image.size:
            mask_image = mask_image.resize(init_image.size, Image.Resampling.LANCZOS)

        w, h = init_image.size
        w, h = (w // 8) * 8, (h // 8) * 8
        init_image = init_image.resize((w, h), Image.Resampling.LANCZOS)
        mask_image = mask_image.resize((w, h), Image.Resampling.LANCZOS)

        pipe = load_pipeline(_model_path)
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

        def progress_callback(pipe, step_index, timestep, callback_kwargs):
            _inpaint_jobs[job_id]["step"] = step_index + 1
            _inpaint_jobs[job_id]["elapsed_sec"] = round(time.time() - start_time, 1)
            return callback_kwargs

        result = pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            negative_prompt=negative_prompt,
            generator=generator,
            callback_on_step_end=progress_callback,
        )

        elapsed = round(time.time() - start_time, 1)
        output = result.images[0]
        buf = io.BytesIO()
        output.save(buf, format="PNG")
        output_b64 = base64.b64encode(buf.getvalue()).decode()

        _inpaint_jobs[job_id] = {
            "status": "done",
            "step": num_steps,
            "total_steps": num_steps,
            "elapsed_sec": elapsed,
            "image": f"data:image/png;base64,{output_b64}",
        }
    except Exception as e:
        _inpaint_jobs[job_id] = {
            "status": "error",
            "error": str(e),
            "elapsed_sec": round(time.time() - start_time, 1),
        }


@app.route("/api/inpaint", methods=["POST"])
def inpaint():
    """Mulai inpainting inference (background job), return job_id untuk polling."""
    data = request.get_json() or {}
    image_b64 = data.get("image")
    mask_b64 = data.get("mask")
    prompt = data.get("prompt", "").strip()
    negative_prompt = data.get("negative_prompt", "blurry, low quality")
    strength = float(data.get("strength", 0.75))
    guidance_scale = float(data.get("guidance_scale", 7.5))
    num_steps = int(data.get("num_steps", 50))
    seed = int(data.get("seed", 42))

    if not image_b64:
        return jsonify({"success": False, "error": "Gambar wajib diupload"}), 400
    if not mask_b64:
        return jsonify({"success": False, "error": "Mask inpainting wajib digambar"}), 400
    if not prompt:
        return jsonify({"success": False, "error": "Prompt wajib diisi"}), 400

    if _pipeline is None:
        return jsonify({"success": False, "error": "Model belum dimuat. Load model terlebih dahulu."}), 400

    job_id = str(uuid.uuid4())
    import threading
    thread = threading.Thread(
        target=_run_inpaint_job,
        args=(job_id, image_b64, mask_b64, prompt, negative_prompt, strength, guidance_scale, num_steps, seed),
    )
    thread.start()
    return jsonify({"success": True, "job_id": job_id})


@app.route("/api/inpaint-status/<job_id>")
def inpaint_status(job_id):
    """Poll progress inpainting."""
    if job_id not in _inpaint_jobs:
        return jsonify({"status": "unknown", "error": "Job tidak ditemukan"}), 404
    return jsonify(_inpaint_jobs[job_id])


@app.route("/api/download-zip", methods=["POST"])
def download_zip():
    """Download hasil crop + mask + inpainting dalam format zip."""
    data = request.get_json() or {}
    image_b64 = data.get("image")
    mask_b64 = data.get("mask")
    result_b64 = data.get("result")

    if not image_b64 or not mask_b64 or not result_b64:
        return jsonify({"success": False, "error": "Data gambar/mask/result tidak lengkap"}), 400

    try:
        def decode_b64(b):
            if "," in b:
                b = b.split(",", 1)[1]
            return base64.b64decode(b)

        image_data = decode_b64(image_b64)
        mask_data = decode_b64(mask_b64)
        result_data = decode_b64(result_b64)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("image.png", image_data)
            zf.writestr("mask.png", mask_data)
            zf.writestr("result.png", result_data)

        buf.seek(0)
        return send_file(
            buf,
            mimetype="application/zip",
            as_attachment=True,
            download_name="inpaint_result.zip",
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/load-zip", methods=["POST"])
def load_zip():
    """Load zip berisi image.png + mask.png, return base64 untuk load ke canvas."""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "File zip wajib diupload"}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".zip"):
        return jsonify({"success": False, "error": "File harus format zip"}), 400

    try:
        with zipfile.ZipFile(file.stream, "r") as zf:
            names = zf.namelist()
            if "image.png" not in names or "mask.png" not in names:
                return jsonify({"success": False, "error": "Zip harus berisi image.png dan mask.png"}), 400

            image_data = zf.read("image.png")
            mask_data = zf.read("mask.png")

        image_b64 = base64.b64encode(image_data).decode()
        mask_b64 = base64.b64encode(mask_data).decode()

        return jsonify({
            "success": True,
            "image": f"data:image/png;base64,{image_b64}",
            "mask": f"data:image/png;base64,{mask_b64}",
        })
    except zipfile.BadZipFile:
        return jsonify({"success": False, "error": "File zip tidak valid"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
