#!/usr/bin/env python
# coding=utf-8
import os
import sys
import subprocess

# Prevent CUDA memory fragmentation under WSL2
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:128"

# Auto-relaunch with accelerate launch if executed directly from VS Code
if "ACCELERATE_TORCH_DEVICE" not in os.environ and os.environ.get("INSIDE_ACCELERATE_RELAUNCH") != "1":
    print("[AUTO-LAUNCH] Detected direct execution. Launching via accelerate with single-GPU fp16 profile...")
    cmd = [
        "accelerate", "launch",
        "--num_processes=1",
        "--mixed_precision=fp16",
        __file__
    ] + sys.argv[1:]
    
    child_env = os.environ.copy()
    child_env["INSIDE_ACCELERATE_RELAUNCH"] = "1"
    result = subprocess.run(cmd, env=child_env)
    sys.exit(result.returncode)

import argparse
import gc
import itertools
import json
import logging
import math
import shutil
import time
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, hf_hub_download, upload_folder
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EDMEulerScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module

logger = get_logger(__name__)

def get_gpu_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        return f"VRAM Alloc: {allocated:.2f} GB | Reserved: {reserved:.2f} GB"
    return "GPU memory stats unavailable"

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"):
    text_encoder_config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder, revision=revision)
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

def parse_args():
    parser = argparse.ArgumentParser(description="SDXL LoRA Training for Custom Face.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--instance_data_dir", type=str, default="dataset/my_face")
    parser.add_argument("--instance_prompt", type=str, default="a photo of ohwx man")
    parser.add_argument("--output_dir", type=str, default="sdxl_lora_face")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=650)
    parser.add_argument("--checkpointing_steps", type=int, default=200)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    return parser.parse_args()

class CachedLatentsDataset(Dataset):
    def __init__(self, latents_list, original_sizes, crop_coords):
        self.latents_list = latents_list
        self.original_sizes = original_sizes
        self.crop_coords = crop_coords

    def __len__(self):
        return len(self.latents_list)

    def __getitem__(self, index):
        return {
            "latent": self.latents_list[index],
            "original_size": self.original_sizes[index],
            "crop_coord": self.crop_coords[index],
        }

def collate_cached(examples):
    return {
        "latents": torch.stack([ex["latent"] for ex in examples]),
        "original_sizes": [ex["original_size"] for ex in examples],
        "crop_coords": [ex["crop_coord"] for ex in examples],
    }

def tokenize_prompt(tokenizer, prompt):
    return tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids

def encode_prompt(text_encoders, tokenizers, prompt):
    prompt_embeds_list = []
    for i, text_encoder in enumerate(text_encoders):
        tokenizer = tokenizers[i]
        text_input_ids = tokenize_prompt(tokenizer, prompt)
        prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False)
        pooled = prompt_embeds[0]
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    return torch.concat(prompt_embeds_list, dim=-1), pooled.view(bs_embed, -1)

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32

    # --- PHASE 1: PRE-ENCODE TEXT PROMPT ---
    logger.info("[PHASE 1] Pre-encoding text prompts...")
    tokenizer_one = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2", use_fast=False)

    text_encoder_cls_one = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2")

    text_encoder_one = text_encoder_cls_one.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", variant=args.variant).to(accelerator.device, dtype=weight_dtype)
    text_encoder_two = text_encoder_cls_two.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2", variant=args.variant).to(accelerator.device, dtype=weight_dtype)

    with torch.no_grad():
        prompt_embeds, unet_add_text_embeds = encode_prompt([text_encoder_one, text_encoder_two], [tokenizer_one, tokenizer_two], args.instance_prompt)
        prompt_embeds = prompt_embeds.to(accelerator.device)
        unet_add_text_embeds = unet_add_text_embeds.to(accelerator.device)

    del tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two
    gc.collect()
    torch.cuda.empty_cache()

    # --- PHASE 2: PRE-ENCODE AND CACHE VAE LATENTS ---
    logger.info("[PHASE 2] Loading images and pre-caching VAE Latents to host RAM (one-time operation)...")
    instance_dir = Path(args.instance_data_dir)
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_paths = [p for p in instance_dir.iterdir() if p.suffix.lower() in valid_exts]

    if not image_paths:
        raise ValueError(f"No valid images found in {instance_dir}")

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", variant=args.variant)
    vae.to("cpu", dtype=torch.float32)
    vae.eval()

    resize_op = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.LANCZOS)
    crop_op = transforms.CenterCrop(args.resolution)
    normalize_op = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    cached_latents = []
    original_sizes = []
    crop_coords = []

    with torch.no_grad():
        for path in tqdm(image_paths, desc="Pre-encoding Latents"):
            img = Image.open(path)
            img = exif_transpose(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            orig_size = (img.height, img.width)
            img = resize_op(img)
            y1 = max(0, int(round((img.height - args.resolution) / 2.0)))
            x1 = max(0, int(round((img.width - args.resolution) / 2.0)))
            img = crop_op(img)
            tensor_img = normalize_op(img).unsqueeze(0).to("cpu", dtype=torch.float32)

            latent = vae.encode(tensor_img).latent_dist.sample() * vae.config.scaling_factor
            cached_latents.append(latent.squeeze(0).to(dtype=weight_dtype))
            original_sizes.append(orig_size)
            crop_coords.append((y1, x1))

    # Free VAE permanently from memory
    del vae
    gc.collect()

    train_dataset = CachedLatentsDataset(cached_latents, original_sizes, crop_coords)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_cached,
        num_workers=0,
    )

    # --- PHASE 3: UNET & OPTIMIZER SETUP ---
    logger.info("[PHASE 3] Loading UNet and configuring LoRA adapters...")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", variant=args.variant)
    unet.requires_grad_(False)
    unet.to(accelerator.device, dtype=weight_dtype)
    unet.enable_gradient_checkpointing()

    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)

    unet_lora_parameters = list(filter(lambda p: p.requires_grad, unet.parameters()))
    cast_training_params([unet], dtype=torch.float32)

    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(
        [{"params": unet_lora_parameters, "lr": args.learning_rate}],
        weight_decay=1e-4,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.max_train_steps,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    def compute_time_ids(original_size, crop_coord):
        target_size = (args.resolution, args.resolution)
        add_time_ids = list(original_size + crop_coord + target_size)
        return torch.tensor([add_time_ids], device=accelerator.device, dtype=weight_dtype)

    logger.info("=" * 60)
    logger.info("             STARTING HIGH-SPEED SDXL TRAINING        ")
    logger.info("=" * 60)
    logger.info(f"  Training Instances       = {len(train_dataset)}")
    logger.info(f"  Resolution               = {args.resolution}x{args.resolution}")
    logger.info(f"  Max Optimization Steps   = {args.max_train_steps}")
    logger.info(f"  Initial VRAM Usage       = {get_gpu_memory_usage()}")
    logger.info("=" * 60)

    global_step = 0
    progress_bar = tqdm(range(0, args.max_train_steps), desc="Steps", disable=not accelerator.is_local_main_process)
    start_time = time.time()
    running_loss = 0.0

    for epoch in range(num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Instant load from cache: zero VAE overhead
                model_input = batch["latents"].to(device=accelerator.device, dtype=weight_dtype)
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device).long()
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                add_time_ids = torch.cat([compute_time_ids(s, c) for s, c in zip(batch["original_sizes"], batch["crop_coords"])])
                unet_added_conditions = {
                    "time_ids": add_time_ids,
                    "text_embeds": unet_add_text_embeds.repeat(bsz, 1),
                }

                model_pred = unet(
                    noisy_model_input,
                    timesteps,
                    prompt_embeds.repeat(bsz, 1, 1),
                    added_cond_kwargs=unet_added_conditions,
                    return_dict=False,
                )[0]

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet_lora_parameters, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                curr_loss = loss.detach().item()
                running_loss += curr_loss

                if global_step % args.log_interval == 0 or global_step == 1:
                    avg_loss = running_loss / (args.log_interval if global_step > 1 else 1)
                    elapsed = time.time() - start_time
                    speed = global_step / elapsed if elapsed > 0 else 0
                    eta_min = ((args.max_train_steps - global_step) / speed) / 60 if speed > 0 else 0

                    logger.info(
                        f"[STEP {global_step:03d}/{args.max_train_steps}] "
                        f"Epoch {epoch + 1:02d}/{num_train_epochs} | "
                        f"Loss: {curr_loss:.4f} (Avg: {avg_loss:.4f}) | "
                        f"Speed: {speed:.2f} it/s | "
                        f"ETA: {eta_min:.1f}m | "
                        f"{get_gpu_memory_usage()}"
                    )
                    running_loss = 0.0

                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"[CHECKPOINT] Saved state to {save_path}")

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("[FINALIZING] Saving LoRA weights...")
        unet_unwrap = accelerator.unwrap_model(unet).to(torch.float32)
        unet_unwrap = unet_unwrap._orig_mod if is_compiled_module(unet_unwrap) else unet_unwrap
        unet_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet_unwrap))
        output_lora_path = os.path.join(args.output_dir, "pytorch_lora_weights.safetensors")
        StableDiffusionXLPipeline.save_lora_weights(save_directory=args.output_dir, unet_lora_layers=unet_lora_layers)
        logger.info(f"[COMPLETED] Weights saved successfully to: {output_lora_path}")

    accelerator.end_training()

if __name__ == "__main__":
    main()