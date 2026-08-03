import numpy as np
import torch
import torchvision.transforms.functional as TF
from diffusers import AutoencoderKLWan, SkyReelsV2ImageToVideoPipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video
from PIL import Image

# 1. Model Configuration - Optimized 1.3B I2V Model for 540P
model_id = "Skywork/SkyReels-V2-I2V-1.3B-540P-Diffusers"

print("Loading VAE and Pipeline...")
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)

pipeline = SkyReelsV2ImageToVideoPipeline.from_pretrained(
    model_id, 
    vae=vae, 
    torch_dtype=torch.bfloat16
)

# Fix for CLIP Image Encoder dtype mismatch
if hasattr(pipeline, "image_encoder") and pipeline.image_encoder is not None:
    pipeline.image_encoder.to(dtype=torch.bfloat16)

# 2. Configure Scheduler (flow_shift = 5.0 is recommended for I2V)
flow_shift = 5.0
pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config, flow_shift=flow_shift)

# 3. Offload components to System RAM to fit within RTX 3060 Ti VRAM
pipeline.enable_model_cpu_offload()

# Enable progress bar
pipeline.set_progress_bar_config(disable=False)

# 4. Load Local Input Image from My-Media directory
image_path1 = "My-Media/flf2v_input_first_frame.png" 
image_path2 = "My-Media/image2.png" 

# Choose an image
image_path = image_path2

print(f"Loading local image from: {image_path}")
first_frame = Image.open(image_path).convert("RGB")

# 5. Image Processing Helper Functions (Targeting 540P max resolution area: 544 x 960)
def aspect_ratio_resize(image, pipeline, max_area=544 * 960):
    aspect_ratio = image.height / image.width
    mod_value = pipeline.vae_scale_factor_spatial * pipeline.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((width, height))
    return image, height, width

first_frame, height, width = aspect_ratio_resize(first_frame, pipeline)

# 6. Text Prompt describing the action
prompt1 = (
    "CG animation style, a small blue bird takes off from the ground, flapping its wings. "
    "The bird's feathers are delicate, with a unique pattern on its chest. "
    "The background shows a blue sky with white clouds under bright sunshine."
)

prompt2 = (
    "The clockwork explorer slowly lifts the crystal lantern, "
    "causing the pulsing violet energy inside it to shine brighter and cast vivid light across their suit. "
    "Dense icy mist gently rolls across the frozen ground. "
    "As the explorer takes a slow, heavy step forward toward the camera, "
    "glowing particles float out of the lantern into the cold night air, "
    "cinematic slow motion, dynamic lighting, ultra-realistic physics."
)


print("Starting Image-to-Video generation...")

# 7. Run Inference
output = pipeline(
    image=first_frame,
    prompt=prompt2,
    num_inference_steps=30,
    height=height,
    width=width,
    guidance_scale=5.0,  # Recommended 5.0 for I2V
    num_frames=97,       # 97 frames (~4 seconds at 24 fps)
).frames[0]

# 8. Save Output Video
export_to_video(output, "i2v_output_1.3B.mp4", fps=24, quality=8)
print("Image-to-Video generated successfully on RTX 3060 Ti!")