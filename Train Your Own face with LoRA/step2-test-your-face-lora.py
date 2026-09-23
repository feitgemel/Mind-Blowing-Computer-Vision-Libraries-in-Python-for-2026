import os
import torch
from pathlib import Path
from datetime import datetime
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# Prevent CUDA memory fragmentation under WSL2
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:128"

# 1. Pipeline configuration
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
lora_dir = "sdxl_lora_face"
lora_weight_name = "pytorch_lora_weights.safetensors"

# Destination folder: 'result'
output_dir = Path("result")
output_dir.mkdir(parents=True, exist_ok=True)

print("[1/4] Loading SDXL Base Pipeline...")
pipeline = StableDiffusionXLPipeline.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)

# Crucial memory optimizations for 8GB GPUs:
pipeline.enable_model_cpu_offload()  # Keeps only active components on GPU
pipeline.enable_vae_tiling()         # Prevents VAE decode OOM crashes

# 2. Optimized scheduler
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
    pipeline.scheduler.config,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True,
)

print("[2/4] Loading trained Face LoRA weights...")
pipeline.load_lora_weights(lora_dir, weight_name=lora_weight_name)

# 3. Prompt setup : The magic word is "ohwx man"
# Use your exact trigger keyword: "ohwx man"

# Prompt1
#prompt = "cinematic 8k portrait of ohwx man in a sharp tailored dark navy suit, soft natural window light, bokeh city background, highly detailed skin texture, 85mm lens photography"
#negative_prompt = "deformed, distorted, disfigured, poorly drawn, bad eyes, blurry, cartoon, 3d render, oversaturated, plastic skin"

# prompt2
#prompt = "close-up portrait of ohwx man wearing a high-tech matte black utility jacket, illuminated by neon turquoise and magenta street signs, rainy night in Tokyo, wet reflections, shallow depth of field, anamorphic lens, blade runner aesthetic, hyper-realistic, 8k"
#negative_prompt = "deformed, distorted, disfigured, poorly drawn, bad eyes, blurry, cartoon, 3d render, oversaturated, plastic skin, drawing, painting"


# prompt3
#prompt = "authentic 1970s vintage color photograph of ohwx man sitting in a classic European café terrace, warm autumn afternoon sunlight, wearing an earth-toned turtleneck sweater, 35mm film grain, Kodachrome color grading, natural candid expression, soft focus background"
#negative_prompt = "modern digital photography, hyper-glossy, plastic skin, CGI, 3d render, cartoon, oversaturated, deformed, distorted, disfigured, bad eyes, blurry, unnatural lighting"

# prompt4
prompt = "cinematic dynamic action shot of ohwx man with modern stylish eyeglasses as Superman flying high in the sky, soaring above the skyscrapers of New York City, wearing clear spectacle frames, classic blue textured suit with iconic chest emblem and flowing red cape, dramatic golden hour sunset illumination, lens flare, motion blur on clouds below, hyper-realistic face details, sharp eyes behind glasses, photorealistic 8k, IMAX cinematic composition"
negative_prompt = "sunglasses, tinted glasses, broken frames, deformed, distorted, disfigured, poorly drawn, bad eyes, blurry, cartoon, 3d render, anime, illustration, green screen effect, toy, figurine, plastic suit, fake cape, oversaturated, bad anatomy, extra limbs"

# Prompt5
#prompt = "masterpiece oil painting of ohwx man with glasses as an Elizabethan noble, dark textured canvas background, rich chiaroscuro dramatic side lighting by Rembrandt, intricate fabric details on velvet collar, authentic oil brush strokes, fine art portrait"
#negative_prompt = "photo, photography, digital photo, modern clothing, 3d render, CGI, smooth plastic, anime, deformed, distorted, disfigured, bad eyes, bad anatomy, blurry"


# Prompt 6
#prompt = "dramatic black and white studio portrait of ohwx man with glasses,age 50, high contrast lighting with deep shadows and sharp highlights, intense gaze, neutral expression, Hasselblad medium format photography, award-winning editorial cover"
#negative_prompt = "color, beard, saturated, sepia, warm tint, colorful, deformed, distorted, disfigured, bad eyes, blurry, 3d render, cartoon, plastic skin, low contrast, washed out"

# Prompt 7

#prompt = "a cinematic 8k portrait of ohwx man in a stylish black leather jacket, soft studio lighting, sharp focus, ultra-detailed face"
#negative_prompt = "deformed, distorted, blurry, bad anatomy, bad eyes, low quality, cartoon, oversaturated"

# 4. Generation settings
num_images = 4
guidance_scale = 6.5
num_inference_steps = 30
lora_scale = 0.95

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"[3/4] Generating {num_images} images sequentially (1-by-1 to stay within 8GB VRAM)...")

for i in range(num_images):
    seed = torch.randint(0, 1000000, (1,)).item()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    print(f"\n--- Generating Image {i + 1}/{num_images} (Seed: {seed}) ---")
    
    with torch.inference_mode():
        image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            cross_attention_kwargs={"scale": lora_scale},
        ).images[0]
        
    file_path = output_dir / f"face_{timestamp}_{i + 1:02d}.png"
    image.save(file_path)
    print(f" -> Successfully saved: {file_path}")

print(f"\n[COMPLETED] All {num_images} images are ready in '{output_dir}/'.")