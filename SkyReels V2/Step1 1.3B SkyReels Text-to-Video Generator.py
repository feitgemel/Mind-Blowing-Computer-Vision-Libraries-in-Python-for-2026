import torch
from diffusers import AutoModel, SkyReelsV2DiffusionForcingPipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video

# Using the much lighter 1.3B model version
model_id = "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers"

# 1. Load VAE and Pipeline
vae = AutoModel.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)

pipeline = SkyReelsV2DiffusionForcingPipeline.from_pretrained(
    model_id,
    vae=vae,
    torch_dtype=torch.bfloat16
)

# 2. Configure Scheduler
flow_shift = 8.0  
pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config, flow_shift=flow_shift)

# 3. Critical for RTX 3060 Ti: Offloads parts of the model to System RAM when not actively in use
pipeline.enable_model_cpu_offload()

# 4. Text Prompt

# Swan 
prompt1 = (
    "A graceful white swan with a curved neck swimming in a serene lake at dawn, "
    "its reflection perfectly mirrored in the still water as mist rises from the surface."
)

# watch melting
prompt2 = (
    "A surreal, "
    "giant ornate pocket watch melting over the edge of a stone balcony in a quiet Venice-style city at sunset."
    "Golden liquid time drips slowly from its hands into the glowing canal below, creating soft concentric ripples."
    "The camera slowly zooms in as warm ambient light catches the intricate bronze gears exposed inside,"
    "high detail, atmospheric, 8k resolution cinematic look."
)


# Mechanical Dragon

prompt3 = (
    "A majestic bioluminescent mechanical dragon, "
    "crafted from dark chrome and glowing cyan neon tubing,"
    "resting atop a snowy skyscraper in Tokyo at night."
    "Falling snowflakes gently melt as they touch its warm glowing body."
    "The dragon slowly lifts its head,"
    "opening bright golden eyes,"
    "while a soft plume of blue steam drifts from its nostrils into the cold night air,"
    "dramatic lighting, volumetric fog."
)


prompt4 = (
    "A massive celestial whale made of translucent blue crystal gliding gracefully through a glowing forest pool filled with floating starlight particles."
    "Deep vibrant purples and emerald greens illuminate the underwater scene."
    "As the whale softly breaches the surface,"
    "liquid light cascades off its back like fine diamonds,"
    "creating shimmering waves in slow motion,"
    "dreamlike composition,"
    "hyper-realistic physics."
)



# The chosen prompt 
prompt = prompt4

# 5. Run Inference (Starting with 257 frames ≈ 10 seconds for testing; can be bumped to 737 for 30 seconds)

# Display progress bar
pipeline.set_progress_bar_config(disable=False)

# The first run is the longest one !!

output = pipeline(
    prompt=prompt,
    num_inference_steps=30,
    height=544,
    width=960,
    base_num_frames=97, 
    num_frames=257,         # Recommended to start with 257 frames (~10s) to verify VRAM headroom
    overlap_history=17,
    addnoise_condition=20,
    ar_step=0,
).frames[0]

# 6. Save Output Video
export_to_video(output, "Output_video_1.3B.mp4", fps=24, quality=8)
print("Video generated successfully on RTX 3060 Ti!")