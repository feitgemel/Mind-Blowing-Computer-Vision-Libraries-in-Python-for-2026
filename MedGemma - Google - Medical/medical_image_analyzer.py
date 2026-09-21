import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# Hugging Face model repository identifier
MODEL_ID = "google/medgemma-4b-it"


def display_image(image_path: str):
    """
    Display the medical image using matplotlib in a non-blocking window.
    """
    img = Image.open(image_path)
    plt.figure(figsize=(7, 7))
    plt.imshow(img)
    plt.title(f"Target: {os.path.basename(image_path)}", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    # Non-blocking show allows inference to run while the image window remains open
    plt.show(block=False)
    plt.pause(0.5)


def load_medgemma(use_4bit: bool = True):
    """
    Load the MedGemma model and processor.
    Uses 4-bit quantization via bitsandbytes to fit comfortably within 8GB VRAM.
    """
    print(f"Loading model checkpoint: {MODEL_ID}...")

    # Configure 4-bit NF4 quantization to minimize GPU VRAM footprint
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    # Load processor handling both image resizing (896x896) and tokenizer chat templates
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # Load multimodal model
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config
    )

    print("Model successfully loaded into memory.")
    return processor, model


def analyze_medical_image(
    image_path: str,
    prompt: str,
    processor: AutoProcessor,
    model: AutoModelForImageTextToText,
    max_new_tokens: int = 400
) -> str:
    """
    Run visual question answering / diagnosis inference on a local medical image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Target image not found at path: {image_path}")

    # Ensure RGB color space for SigLIP encoder
    raw_image = Image.open(image_path).convert("RGB")

    # Construct the multimodal conversation structure expected by Gemma 3 / MedGemma
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are an expert medical specialist and clinical diagnostic assistant. "
                        "Thoroughly analyze the provided medical image across its relevant modality "
                        "(radiology, dermatology, ophthalmology, or pathology). "
                        "Describe the key anatomical and clinical findings, and clearly specify "
                        "whether any pathology, abnormality, lesion, or disease is detected."
                    )
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": raw_image}
            ]
        }
    ]

    # Preprocess both text and image using the chat template
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    # Deterministic generation for factual clinical consistency
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        generated_tokens = outputs[0][input_len:]

    # Decode tokens back into readable text
    return processor.decode(generated_tokens, skip_special_tokens=True)


if __name__ == "__main__":
    # Path to local sample file
    
    # test_image_path = "Normal chest radiograph - female.jpeg"
    # test_image_path = "melanoma.jpg"
    #test_image_path = "Diabetic Retinopathy.jpg"
    test_image_path = "carcinoma.jpg"

    if os.path.exists(test_image_path):
        # 1. Display the selected image window
        print(f"\nDisplaying image: {test_image_path}")
        display_image(test_image_path)

        # 2. Initialize the model in 4-bit mode (consumes ~3-4GB VRAM)
        processor, model = load_medgemma(use_4bit=True)

        # Clinical inquiry prompt
        clinical_query = (
            "Analyze this medical image: "
            "1. Identify the anatomical modality and perspective. "
            "2. Are there any visible abnormalities, opacities, fractures, or pathologies? "
            "3. Provide a concluding clinical diagnostic impression."
        )

        print(f"\nAnalyzing: {test_image_path}...")
        report = analyze_medical_image(test_image_path, clinical_query, processor, model)
        print("\n--- MedGemma Analysis Report ---")
        print(report)

        # Keep the matplotlib plot open until manually closed by the user
        print("\nClose the image window to finish execution.")
        plt.show()
    else:
        print(f"\nImage file '{test_image_path}' was not found. Place a sample image in the path and rerun.")