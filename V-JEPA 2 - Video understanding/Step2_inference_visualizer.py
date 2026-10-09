import os
import glob
import cv2
import numpy as np
import torch
from torchcodec.decoders import VideoDecoder
from transformers import AutoModelForVideoClassification, AutoVideoProcessor


def draw_predictions_overlay(frame: np.ndarray, predictions: list, video_name: str) -> np.ndarray:
    """Draw a semi-transparent HUD overlay with clear, readable fonts."""
    overlay = frame.copy()
    height, width = frame.shape[:2]

    # Proportional scaling calibrated for display size
    scale_factor = max(width / 1280.0, 0.85)
    font_scale_title = 0.75 * scale_factor
    font_scale_text = 0.65 * scale_factor
    line_spacing = int(34 * scale_factor)

    box_width = min(int(width * 0.85), int(720 * scale_factor))
    box_height = int(50 * scale_factor) + len(predictions) * line_spacing
    top_left = (20, 20)
    bottom_right = (20 + box_width, 20 + box_height)

    # Semi-transparent dark background for contrast
    cv2.rectangle(overlay, top_left, bottom_right, (15, 15, 15), -1)
    alpha = 0.82
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Header title
    title = f"V-JEPA 2 | {video_name}"
    title_y = int(45 * scale_factor)
    cv2.putText(
        frame,
        title,
        (35, title_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale_title,
        (0, 255, 255),
        max(int(2 * scale_factor), 2),
        cv2.LINE_AA,
    )

    cv2.line(
        frame,
        (35, title_y + 10),
        (bottom_right[0] - 20, title_y + 10),
        (90, 90, 90),
        1,
    )

    # Draw predicted actions
    y_offset = title_y + int(40 * scale_factor)
    for rank, (action, pct) in enumerate(predictions, start=1):
        # Top-1 in bright green with bold thickness
        if rank == 1:
            color = (0, 255, 100)
            thickness = max(int(2 * scale_factor), 2)
        else:
            color = (235, 235, 235)
            thickness = 1

        line_text = f"#{rank}  {action[:40]}  ({pct:.1f}%)"
        cv2.putText(
            frame,
            line_text,
            (35, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale_text,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y_offset += line_spacing

    return frame


def play_and_save_visualized_video(
    video_path: str, predictions: list, output_dir: str = "results", scale: float = 0.60
):
    """Downscale video to target scale, display with HUD overlay, and save into results/ folder."""
    # Ensure results directory exists
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(video_path)
    output_path = os.path.join(output_dir, f"annotated_{base_name}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Could not open video for visualization: {video_path}")
        return

    # Extract original dimensions
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    delay = int(1000 / fps)

    # Compute target resolution based on scale
    target_w = int(orig_w * scale)
    target_h = int(orig_h * scale)

    print(f"[*] Original resolution: {orig_w}x{orig_h}")
    print(f"[*] Rescaling to {int(scale * 100)}%: {target_w}x{target_h}")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))

    print(f"[*] Displaying: {base_name} (Press 'q' to skip to next video)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame
        frame = cv2.resize(
            frame, (target_w, target_h), interpolation=cv2.INTER_AREA
        )

        # Draw HUD overlay on the resized frame
        annotated_frame = draw_predictions_overlay(
            frame, predictions, base_name
        )
        out.write(annotated_frame)

        # Display window
        cv2.imshow("V-JEPA 2 Action Classifier", annotated_frame)
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            print("[*] Playback interrupted by user.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[+] Saved annotated video to: {output_path}")


def run_video_classification(video_path: str, model, processor, device: str):
    """Run inference on a single local video file and trigger visualization."""
    print(f"\n[*] Processing video: {video_path}")

    try:
        decoder = VideoDecoder(video_path)
    except Exception as e:
        print(f"[!] Error decoding {video_path}: {e}")
        return

    total_frames = len(decoder)
    num_required_frames = getattr(model.config, "frames_per_clip", 16)

    if total_frames < num_required_frames:
        print(
            f"[!] Warning: Video has only {total_frames} frames (required: {num_required_frames}). Skipping."
        )
        return

    # Uniformly sample 16 frames across the entire video
    frame_indices = np.linspace(
        0, total_frames - 1, num_required_frames, dtype=int
    )
    video_tensor = decoder.get_frames_at(
        indices=frame_indices
    ).data  # Shape: (T, C, H, W)

    # Preprocess frames and prepare tensors for model input
    inputs = processor(video_tensor, return_tensors="pt").to(device)

    # Execute forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Compute softmax probabilities and extract top-5 actions
    probabilities = torch.softmax(logits, dim=-1)[0]
    top5_probs, top5_indices = torch.topk(probabilities, 5)

    predictions = []
    print("-" * 55)
    print(f"Top 5 Predicted Actions for: {os.path.basename(video_path)}")
    print("-" * 55)
    for rank, (idx, prob) in enumerate(zip(top5_indices, top5_probs), start=1):
        class_name = model.config.id2label[idx.item()]
        percentage = prob.item() * 100
        predictions.append((class_name, percentage))
        print(f"{rank}. {class_name:<40} | {percentage:5.1f}%")
    print("-" * 55)

    # Save to the 'results' folder and display at 45% of original dimensions
    play_and_save_visualized_video(
        video_path, predictions, output_dir="results", scale=0.40
    )


def main():
    VIDEOS_DIR = "videos"
    SUPPORTED_EXTENSIONS = ("*.mp4", "*.avi", "*.mov", "*.mkv")

    if not os.path.exists(VIDEOS_DIR):
        os.makedirs(VIDEOS_DIR)
        print(f"[*] Directory '{VIDEOS_DIR}' was not found, so it was created.")
        print(f"[*] Please place your video files into '{VIDEOS_DIR}/' and re-run.")
        return

    video_files = []
    for ext in SUPPORTED_EXTENSIONS:
        video_files.extend(glob.glob(os.path.join(VIDEOS_DIR, ext)))

    if not video_files:
        print(f"[!] No video files found in '{VIDEOS_DIR}/'.")
        print(f"[*] Place your .mp4 / .mov clips inside '{VIDEOS_DIR}/' and try again.")
        return

    print(f"[*] Found {len(video_files)} video(s) to evaluate.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Using device: {device}")

    model_id = "facebook/vjepa2-vitl-fpc16-256-ssv2"
    print(f"[*] Loading model: {model_id} ...")
    model = AutoModelForVideoClassification.from_pretrained(model_id).to(device)
    processor = AutoVideoProcessor.from_pretrained(model_id)
    print("[+] Model loaded successfully.")

    for video_file in sorted(video_files):
        run_video_classification(video_file, model, processor, device)


if __name__ == "__main__":
    main()