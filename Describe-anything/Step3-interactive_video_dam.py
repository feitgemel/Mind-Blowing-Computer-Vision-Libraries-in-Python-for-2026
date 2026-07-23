
import torch
import numpy as np
from PIL import Image
import cv2
import sys
import os
import tempfile
import shutil

# Inject project root path to allow seamless imports from any working directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dam import DescribeAnythingModel, disable_torch_init
from sam2.build_sam import build_sam2_video_predictor

# ==================== CONFIGURATION VARIABLES ====================
VIDEO_PATH = "My-Media/Lilach.mp4"              # Path to your input video file
MODEL_PATH = "nvidia/DAM-3B-Video"       # Video-specific DAM checkpoint
PROMPT_MODE = "focal_prompt"
CONV_MODE = "v1"
TEMPERATURE = 0.2
TOP_P = 0.5

SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
# =================================================================

# Global variables for tracking multiple points on the first frame
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

def extract_frames_from_video(video_path):
    """Extract all frames from a video file and save them to a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(temp_dir, f"{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        frame_count += 1
    
    cap.release()
    if frame_count == 0:
        raise ValueError("No frames were extracted from the video. Check the file path.")
    
    return frame_paths, temp_dir

def apply_sam2_video(image_files, points):
    """Initialize SAM2 state and propagate point tracking throughout the video frames."""
    video_dir = os.path.dirname(image_files[0])
    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)

    ann_frame_idx = 0  # First frame index
    ann_obj_id = 1     # Tracking Object ID

    with torch.autocast("cuda", dtype=torch.bfloat16):
        # Convert points to numpy array and add labels (1 = foreground)
        points_np = np.array(points, dtype=np.float32)
        labels_np = np.ones(len(points_np), dtype=np.int32)
        
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points_np,
            labels=labels_np
        )

        # Propagate through the video and collect generated masks
        masks = []
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            mask = (out_mask_logits[0] > 0.0).cpu().numpy()
            masks.append(mask)

    return masks

def print_streaming(text):
    print(text, end="", flush=True)

# Optimization parameters for newer GPU architectures (Ampere or newer)
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Extract frames from video source
print(f"Extracting frames from video: {VIDEO_PATH}...")
image_files, temp_frame_dir = extract_frames_from_video(VIDEO_PATH)

# Read the first frame for the interactive GUI selection
img_cv_clean = cv2.imread(image_files[0])
img_cv_show = img_cv_clean.copy()

# 2. Setup OpenCV interactive window to collect seed points
window_name = "Select Points on First Frame - Press ENTER when done"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, click_event)

print("\n--> INSTRUCTIONS:")
print("1. Click anywhere on the image to add multiple foreground points targeting the object.")
print("2. Press ENTER or SPACEBAR when you are done selecting points to run video propagation.")
print("3. Press ESC to cancel.")

while True:
    cv2.imshow(window_name, img_cv_show)
    key = cv2.waitKey(1) & 0xFF
    if (key == 13 or key == 32) and len(clicked_points) > 0:
        break
    if key == 27:
        print("Execution cancelled.")
        cv2.destroyAllWindows()
        shutil.rmtree(temp_frame_dir)
        exit()

cv2.destroyAllWindows()

# 3. Load SAM2 and DAM video models on the target device
print(f"\nLoading SAM2 and DAM models to {device}...")
predictor = build_sam2_video_predictor(MODEL_CFG, SAM2_CHECKPOINT, device=device)

disable_torch_init()
prompt_modes = {"focal_prompt": "full+focal_crop"}
dam = DescribeAnythingModel(
    model_path=MODEL_PATH,
    conv_mode=CONV_MODE,
    prompt_mode=prompt_modes.get(PROMPT_MODE, PROMPT_MODE),
).to(device)

# 4. Process entire video sequence using SAM2 tracking
print(f"Propagating {len(clicked_points)} tracking points across all frames...")
all_video_masks = apply_sam2_video(image_files, points=clicked_points)

# 5. Uniformly select 8 frames and corresponding masks across the sequence for DAM input
indices = np.linspace(0, len(image_files) - 1, 8, dtype=int)
selected_files = [image_files[i] for i in indices]
selected_masks = [all_video_masks[i] for i in indices]

processed_images = [Image.open(f).convert('RGB') for f in selected_files]
processed_masks = [Image.fromarray((m.squeeze() * 255).astype(np.uint8)) for m in selected_masks]

# 6. Generate and stream localized video description
query = (
    "Video: <image><image><image><image><image><image><image><image>\n"
    "Given the video in the form of a sequence of frames above, "
    "describe the object in the masked region in the video in detail."
)

print("\nDescription:")
for token in dam.get_description(
        processed_images,
        processed_masks,
        query,
        streaming=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        num_beams=1,
        max_new_tokens=512
):
    print_streaming(token)

# Cleanup temporary files
shutil.rmtree(temp_frame_dir)
print("\n\nDone.")