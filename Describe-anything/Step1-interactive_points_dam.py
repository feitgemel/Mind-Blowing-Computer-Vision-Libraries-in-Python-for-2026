import ast
import torch
import numpy as np
from PIL import Image
from transformers import SamModel, SamProcessor
import cv2
import sys
import os

# Inject project root path to allow seamless imports from any working directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dam import DescribeAnythingModel, disable_torch_init

# ==================== CONFIGURATION VARIABLES ====================
IMAGE_PATH = "My-Media/Sunflower-couple2.jpg"
MODEL_PATH = "nvidia/DAM-3B"
PROMPT_MODE = "focal_prompt"
CONV_MODE = "v1"
TEMPERATURE = 0.2
TOP_P = 0.5
# =================================================================

# Global variables for tracking multiple points
clicked_points = []

def click_event(event, x, y, flags, param):
    global clicked_points, img_cv_show

    if event == cv2.EVENT_LBUTTONDOWN:
        # Save the new point coordinates
        clicked_points.append([x, y])
        
        # Draw a prominent large filled circle with a border
        cv2.circle(img_cv_show, (x, y), radius=8, color=(0, 255, 0), thickness=-1)
        cv2.circle(img_cv_show, (x, y), radius=9, color=(0, 0, 0), thickness=1)
        
        # Display the updated image with the counter of points
        cv2.putText(img_cv_show, f"P{len(clicked_points)}", (x + 12, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def apply_sam(image, input_points=None, input_labels=None):
    inputs = sam_processor(image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = sam_model(**inputs)

    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )[0][0]
    scores = outputs.iou_scores[0, 0]
    mask_selection_index = scores.argmax()
    return masks[mask_selection_index].numpy()

def print_streaming(text):
    print(text, end="", flush=True)

# Load image for GUI visualization and model inference
img_pil = Image.open(IMAGE_PATH).convert('RGB')
img_cv_clean = cv2.imread(IMAGE_PATH)
img_cv_show = img_cv_clean.copy()

# Initialize vision models on target device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading SAM and DAM models to {device}...")
sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

disable_torch_init()
prompt_modes = {"focal_prompt": "full+focal_crop"}
dam = DescribeAnythingModel(
    model_path=MODEL_PATH,
    conv_mode=CONV_MODE,
    prompt_mode=prompt_modes.get(PROMPT_MODE, PROMPT_MODE),
).to(device)

# Setup OpenCV window and bind mouse callback for points selection
window_name = "Select Points - Press ENTER to Process"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, click_event)

print("\n--> INSTRUCTIONS:")
print("1. Click anywhere on the image to add multiple foreground points targeting the object.")
print("2. Press ENTER or SPACEBAR when you are done selecting points to run inference.")
print("3. Press ESC to cancel.")

while True:
    cv2.imshow(window_name, img_cv_show)
    key = cv2.waitKey(1) & 0xFF
    
    # Check if Enter (13) or Spacebar (32) is pressed to finish selection
    if (key == 13 or key == 32) and len(clicked_points) > 0:
        break
    # Cancel execution safely if ESC (27) is pressed
    if key == 27:
        print("Execution cancelled.")
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()

# Format points and labels for Segment Anything Model
# SAM expects input_points shape: [batch_size, num_points, 2]
input_points = [clicked_points]
# Label 1 indicates a foreground point
input_labels = [[1] * len(clicked_points)]

print(f"\nProcessing matching mask using {len(clicked_points)} points: {clicked_points}")

# Generate localized mask via SAM using multiple points
mask_np = apply_sam(img_pil, input_points=input_points, input_labels=input_labels)
mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))

# Generate and stream multimodal text description
query = "<image>\nDescribe the masked region in detail."
print("\nDescription:")

for token in dam.get_description(
        img_pil,
        mask_pil,
        query,
        streaming=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        num_beams=1,
        max_new_tokens=512
):
    print_streaming(token)
print("\n\nDone.")