import os
import av
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM

# ==========================================
# CONFIGURATION
# ==========================================
#HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX"
HF_TOKEN = os.getenv("HF_TOKEN", "YOUR_HUGGING_FACE_TOKEN")

MODEL_ID = "NemoStation/Marlin-2B"
VIDEO_PATH = "videos/security_camera.mp4"
OUTPUT_GRID_PATH = "detected_event_15_frames.jpg"

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchcodec"
os.environ["VIDEO_MAX_PIXELS"] = "200704"  # Frame resolution ~448x448
os.environ["FPS"] = "2.0"                  # Sample rate: 2 frames per second
os.environ["FPS_MAX_FRAMES"] = "240"       # Default model cap (covers ~2 minutes)


if not os.path.exists(VIDEO_PATH):
    print(f"Error: Target video file '{VIDEO_PATH}' not found. Please provide a valid file.")
    exit()

print("Loading model weights to GPU (bfloat16)...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map={"": "cuda"},
    token=HF_TOKEN
)

# Find mode : Detect specific events in the video by query 
# =====================================================================
print("\n" + "=" * 60)
print("MODE 1: Temporal Grounding (Find Mode)")
print("=" * 60)

# Targeted query for close person movement
query = "a person walking close toward the camera"
print(f"Searching for event: '{query}'...")

find_result = model.find(VIDEO_PATH, event=query)


# Only textual output if no event is detected, otherwise extract frames and create a visual grid
#if find_result.get("span"):
#    start_sec, end_sec = find_result["span"]
#    print(f"[ALERT] Person detected between: {start_sec:.1f}s and {end_sec:.1f}s")
#    print(f"Raw model response: {find_result.get('raw')}")
#else:
#    print("[INFO] No matching person activity detected in the specified range.")



if find_result.get("span"):
    start_sec, end_sec = find_result["span"]
    print(f"[ALERT] Person detected between: {start_sec:.1f}s and {end_sec:.1f}s")
    print(f"Raw model response: {find_result.get('raw')}")

    # Extract 15 frames across a 6-second window (1.5s before start_sec)
    window_start = max(0.0, start_sec - 1.5)
    duration = 6.0
    num_frames = 15
    target_timestamps = [window_start + i * (duration / (num_frames - 1)) for i in range(num_frames)]

    container = av.open(VIDEO_PATH)
    extracted_frames = []
    current_target_idx = 0

    # Sequential decode from start guarantees exact timestamp accuracy
    for frame in container.decode(video=0):
        if current_target_idx >= len(target_timestamps):
            break
        if frame.time is not None and frame.time >= target_timestamps[current_target_idx]:
            extracted_frames.append((frame.time, frame.to_image()))
            current_target_idx += 1

    container.close()

    if extracted_frames:
        # Create a 3x5 visual grid (15 frames total)
        fig, axes = plt.subplots(3, 5, figsize=(20, 10))
        axes = axes.flatten()

        for idx, ax in enumerate(axes):
            if idx < len(extracted_frames):
                ts, img = extracted_frames[idx]
                ax.imshow(img)
                ax.set_title(f"{ts:.2f}s", fontsize=11, fontweight="bold")
            ax.axis("off")

        plt.suptitle(
            f"15-Frame Sequence Around Detection ({start_sec:.1f}s - {end_sec:.1f}s)",
            fontsize=15,
            fontweight="bold"
        )
        plt.tight_layout()
        
        plt.savefig(OUTPUT_GRID_PATH, dpi=150)
        print(f"[INFO] Saved 15-frame sequence to '{OUTPUT_GRID_PATH}'")
        plt.show()
else:
    print("[INFO] No matching person activity detected in the specified range.")


# Caption mode : Generate a full scene and event breakdown of the video
# =====================================================================
print("\n" + "=" * 60)
print("MODE 2: Full Scene & Event Breakdown (Caption Mode)")
print("=" * 60)

caption_result = model.caption(VIDEO_PATH, max_new_tokens=2048)

print("\nGeneral Scene Overview:")
print(caption_result.get("scene", "No scene summary available."))

print("\nDetailed Event Timeline:")
events = caption_result.get("events", [])
if events:
    for ev in events:
        print(f"[{ev['start']:.1f}s -> {ev['end']:.1f}s]: {ev['description']}")
else:
    print(caption_result.get("caption", "No events parsed."))