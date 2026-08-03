import torch
import gc
from diffusers import AutoModel, SkyReelsV2DiffusionForcingPipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video

# 1. Model ID - 14B Variant
model_id = "Skywork/SkyReels-V2-DF-14B-540P-Diffusers"

# Free up unallocated CUDA memory
torch.cuda.empty_cache()
gc.collect()

print("Loading VAE (14B)...")
vae = AutoModel.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)

print("Loading Pipeline (14B)...")
pipeline = SkyReelsV2DiffusionForcingPipeline.from_pretrained(
    model_id,
    vae=vae,
    torch_dtype=torch.bfloat16
)

# 2. Configure Scheduler
flow_shift = 8.0  
pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config, flow_shift=flow_shift)

# 3. Offload layer-by-layer to System RAM
pipeline.enable_sequential_cpu_offload()

# Enable VAE Tiling to help with final decode memory
if hasattr(pipeline.vae, "enable_tiling"):
    pipeline.vae.enable_tiling()

# 4. Text Prompt
prompt = (
    "A graceful white swan with a curved neck swimming in a serene lake at dawn, "
    "its reflection perfectly mirrored in the still water as mist rises from the surface."
)

pipeline.set_progress_bar_config(disable=False)

print("Starting 14B inference with full original parameters...")

# 5. Full Original Parameters
output = pipeline(
    prompt=prompt,
    num_inference_steps=30,
    height=544,
    width=960,
    base_num_frames=97,     # Original parameter
    num_frames=257,         # Original parameter (~10 seconds)
    overlap_history=17,
    addnoise_condition=20,
    ar_step=0,
).frames[0]

# 6. Export Output Video
export_to_video(output, "dragon_14B_full.mp4", fps=24, quality=8)
print("Video generated successfully!")