import os
import json
import urllib.request
import torch
import torchaudio
from einops import rearrange

from audiox import get_pretrained_model
from audiox.models.factory import create_model_from_config
from audiox.inference.generation import generate_diffusion_cond
from audiox.data.utils import (
    read_video, 
    merge_video_audio, 
    load_and_process_audio, 
    encode_video_with_synchformer
)

# -------------------------------------------------------------------------
# ==========================================
# 1. Internal Parameters & Configuration
# ==========================================
# Choose model variant: "AudioX-MAF" or "AudioX-MAF-MMDiT"
SELECTED_MODEL = "AudioX-MAF-MMDiT"
# -------------------------------------------------------------------------


# Input definitions
video_path = "example/V2A_sample-1.mp4" 
text_prompt = "Generate music for the video" 
audio_path = None
steps = 250

# Dynamic paths and file names based on SELECTED_MODEL
model_name_full = f"HKUSTAudio/{SELECTED_MODEL}"
local_model_dir = os.path.join("model", SELECTED_MODEL)
local_config_path = os.path.join(local_model_dir, "config.json")
local_ckpt_path = os.path.join(local_model_dir, "model.ckpt")

output_wav_path = f"output_{SELECTED_MODEL}.wav"
output_mp4_path = f"output_{SELECTED_MODEL}.mp4"

# ==========================================
# 2. Auxiliary Dependency Checker
# ==========================================
def ensure_auxiliary_dependencies():
    """Download missing hardcoded model components (like VAE) if missing."""
    os.makedirs("model", exist_ok=True)
    deps = {
        "model/VAE.ckpt": "https://huggingface.co/HKUSTAudio/AudioX-MAF-MMDiT/resolve/main/VAE.ckpt",
        "model/synchformer_state_dict.pth": "https://huggingface.co/HKUSTAudio/AudioX-MAF/resolve/main/synchformer_state_dict.pth"
    }
    for path, url in deps.items():
        if not os.path.exists(path):
            print(f"--> [Auto-Download] Downloading missing dependency: {path}...", flush=True)
            urllib.request.urlretrieve(url, path)
            print(f"--> [Auto-Download] Download completed: {path}", flush=True)

# Ensure required files exist before initializing model
ensure_auxiliary_dependencies()

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--> Using device: {device.upper()}", flush=True)
print(f"--> Selected Model Target: {SELECTED_MODEL}", flush=True)

# ==========================================
# 3. Hybrid Model Loader (Local Check -> Remote Fallback)
# ==========================================
has_local_files = os.path.exists(local_config_path) and os.path.exists(local_ckpt_path)

if has_local_files:
    print(f"--> Loading local model files from: {local_model_dir}...", flush=True)
    with open(local_config_path, "r") as f:
        model_config = json.load(f)

    model = create_model_from_config(model_config)
    ckpt = torch.load(local_ckpt_path, map_location="cpu")
    
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    print("--> Local model loaded successfully.", flush=True)
else:
    print(f"--> Local files not found at '{local_model_dir}'. Falling back to Hugging Face cache...", flush=True)
    model, model_config = get_pretrained_model(model_name_full)
    print("--> Remote model loaded successfully.", flush=True)

# Extract parameters from config
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]
target_fps = model_config["video_fps"]
seconds_start = 0
seconds_total = 10

model = model.to(device)

# ==========================================
# 4. Input Processing
# ==========================================
print(f"--> Reading video frames from: {video_path}", flush=True)
video_tensor = read_video(
    video_path, 
    seek_time=seconds_start, 
    duration=seconds_total, 
    target_fps=target_fps
)

if audio_path:
    print(f"--> Loading guide audio from: {audio_path}", flush=True)
    audio_tensor = load_and_process_audio(audio_path, sample_rate, seconds_start, seconds_total)
else:
    print("--> No input audio provided. Initializing zero tensor placeholder.", flush=True)
    audio_tensor = torch.zeros((2, int(sample_rate * seconds_total)))

# ==========================================
# 5. Extract SynchFormer Features (MAF Models)
# ==========================================
video_sync_frames = None
if "MAF" in SELECTED_MODEL:
    print(f"--> Encoding temporal features with SynchFormer ({model_name_full})...", flush=True)
    video_sync_frames = encode_video_with_synchformer(
        video_path, model_name_full, seconds_start, seconds_total, device
    )

# ==========================================
# 6. Prepare Conditioning Pipeline
# ==========================================
conditioning = [{
    "video_prompt": {
        "video_tensors": video_tensor.unsqueeze(0), 
        "video_sync_frames": video_sync_frames
    },        
    "text_prompt": text_prompt,
    "audio_prompt": audio_tensor.unsqueeze(0),
    "seconds_start": seconds_start,
    "seconds_total": seconds_total
}]

# ==========================================
# 7. Audio Generation (Diffusion Process)
# ==========================================
print(f"--> Starting audio generation ({steps} steps)...", flush=True)
output = generate_diffusion_cond(
    model,
    steps=steps,
    cfg_scale=7,
    conditioning=conditioning,
    sample_size=sample_size,
    sigma_min=0.3,
    sigma_max=500,
    sampler_type="dpmpp-3m-sde",
    device=device,
)

# ==========================================
# 8. Post-Processing & Audio Export
# ==========================================
print("--> Processing generated audio tensor...", flush=True)
output = rearrange(output, "b d n -> d (b n)")
output = (
    output.to(torch.float32)
    .div(torch.max(torch.abs(output)))
    .clamp(-1, 1)
    .mul(32767)
    .to(torch.int16)
    .cpu()
)

torchaudio.save(output_wav_path, output, sample_rate)
print(f"--> Audio exported successfully to: {output_wav_path}", flush=True)

# ==========================================
# 9. Merge Generated Audio with Original Video
# ==========================================
if video_path is not None and os.path.exists(video_path):
    print("--> Merging generated audio with original video...", flush=True)
    merge_video_audio(video_path, output_wav_path, output_mp4_path, seconds_start, seconds_total)
    print(f"--> Final video saved to: {output_mp4_path}", flush=True)

print("--> Process completed successfully!", flush=True)