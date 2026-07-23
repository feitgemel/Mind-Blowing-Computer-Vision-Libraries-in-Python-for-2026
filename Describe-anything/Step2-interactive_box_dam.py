
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

# Global variables for tracking bounding box selection
drawing = False
ix, iy = -1, -1
x1, y1, x2, y2 = -1, -1, -1, -1
box_selected = False

def draw_bounding_box(event, x, y, flags, param):
    global ix, iy, x1, y1, x2, y2, drawing, box_selected, img_cv_show

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Refresh view with clean image and draw current temporary rectangle
            img_cv_show = img_cv_clean.copy()
            cv2.rectangle(img_cv_show, (ix, iy), (x, y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1, x2, y2 = ix, iy, x, y
        box_selected = True
        # Finalize and draw the bounding box rectangle
        cv2.rectangle(img_cv_show, (x1, y1), (x2, y2), (0, 255, 0), 2)

def apply_sam(image, input_boxes=None):
    inputs = sam_processor(image, input_boxes=input_boxes, return_tensors="pt").to(device)

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

# Setup OpenCV window and bind mouse callback for bounding box selection
window_name = "Select Bounding Box - Press ENTER to Process"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, draw_bounding_box)

print("\n--> INSTRUCTIONS:")
print("1. Click and drag with the MOUSE to draw a bounding box around the target object.")
print("2. Press ENTER or SPACEBAR when you are done selecting the box to run inference.")
print("3. Press ESC to cancel.")

while True:
    cv2.imshow(window_name, img_cv_show)
    key = cv2.waitKey(1) & 0xFF
    
    # Check if Enter (13) or Spacebar (32) is pressed to finish selection
    if (key == 13 or key == 32) and box_selected:
        break
    # Cancel execution safely if ESC (27) is pressed
    if key == 27:
        print("Execution cancelled.")
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()

# Format bounding box coordinates for Segment Anything Model
# Expected box coordinates layout: [x_min, y_min, x_max, y_max]
box_coords = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
input_boxes = [[box_coords]]

print(f"\nProcessing matching mask using bounding box: {box_coords}")

# Generate localized mask via SAM using bounding box configuration
mask_np = apply_sam(img_pil, input_boxes=input_boxes)
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