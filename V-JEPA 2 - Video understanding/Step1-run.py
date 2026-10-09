import numpy as np
import torch
from torchcodec.decoders import VideoDecoder
from transformers import AutoModelForVideoClassification, AutoVideoProcessor

# Video file path and compute device configuration
VIDEO_FILE = "videos/Pasta.mp4"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# 1. Load model and processor from Hugging Face
model_id = "facebook/vjepa2-vitl-fpc16-256-ssv2"
print("Loading model...")
model = AutoModelForVideoClassification.from_pretrained(model_id).to(device)
processor = AutoVideoProcessor.from_pretrained(model_id)

# 2. Decode video and uniformly sample 16 frames across its duration
print(f"Reading: {VIDEO_FILE}")
decoder = VideoDecoder(VIDEO_FILE)
indices = np.linspace(0, len(decoder) - 1, 16, dtype=int)
video_tensor = decoder.get_frames_at(indices=indices).data  # (T, C, H, W)

# 3. Preprocess frames and run forward pass
inputs = processor(video_tensor, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

# 4. Compute probabilities and extract the top-5 predicted actions
probs = torch.softmax(outputs.logits, dim=-1)[0]
top_probs, top_indices = torch.topk(probs, 5)

print("\n--------------------------------")
print("\n--- Top 5 Action Predictions ---")
for rank, (idx, prob) in enumerate(zip(top_indices, top_probs), start=1):
    action_name = model.config.id2label[idx.item()]
    print(f"#{rank} {action_name} - {prob.item() * 100:.1f}%")