import functools, torch, os, tqdm
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
apply_liger_kernel_to_qwen2_vl() # important. our model is trained with this. keep consistency
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, LogitsProcessor, logging
from livecc_utils import prepare_multiturn_multimodal_inputs_for_generation, get_smart_resized_clip, get_smart_resized_video_reader
from qwen_vl_utils import process_vision_info

class LiveCCDemoInfer:
    fps = 2
    initial_fps_frames = 6
    streaming_fps_frames = 2
    initial_time_interval = initial_fps_frames / fps
    streaming_time_interval = streaming_fps_frames / fps
    frame_time_interval = 1 / fps
    
    def __init__(self, model_path: str = None, device_id: int = 0):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", 
            device_map=f'cuda:{device_id}', 
            attn_implementation='flash_attention_2'
        )
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
        self.model.prepare_inputs_for_generation = functools.partial(prepare_multiturn_multimodal_inputs_for_generation, self.model)
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": 'livecc'},
            ]
        }
        texts = self.processor.apply_chat_template([message], tokenize=False)
        self.system_prompt_offset = texts.index('<|im_start|>user')
        self._cached_video_readers_with_hw = {}

    def live_cc(
        self,
        query: str,
        state: dict,
        max_pixels: int = 384 * 28 * 28,
        default_query: str = 'Please describe the video.',
        do_sample: bool = False,
        repetition_penalty: float = 1.05,
        **kwargs,
    ): 
        """
        state: dict, (maybe) with keys:
            video_path: str, video path
            video_timestamp: float, current video timestamp
            last_timestamp: float, last processed video timestamp
            last_video_pts_index: int, last processed video frame index
            video_pts: np.ndarray, video pts
            last_history: list, last processed history
            past_key_values: llm past_key_values
            past_ids: past generated ids
        """
        # 1. preparation: video_reader, and last processing info
        video_timestamp, last_timestamp = state.get('video_timestamp', 0), state.get('last_timestamp', -1 / self.fps)
        video_path = state['video_path']
        if video_path not in self._cached_video_readers_with_hw:
            self._cached_video_readers_with_hw[video_path] = get_smart_resized_video_reader(video_path, max_pixels)
            video_reader = self._cached_video_readers_with_hw[video_path][0]
            video_reader.get_frame_timestamp(0)
            state['video_pts'] = torch.from_numpy(video_reader._frame_pts[:, 1])
            state['last_video_pts_index'] = -1
        video_pts = state['video_pts']
        if last_timestamp + self.frame_time_interval > video_pts[-1]:
            state['video_end'] = True
            return 
        video_reader, resized_height, resized_width = self._cached_video_readers_with_hw[video_path]
        last_video_pts_index = state['last_video_pts_index']

        # 2. which frames will be processed
        initialized = last_timestamp >= 0
        if not initialized:
            video_timestamp = max(video_timestamp, self.initial_time_interval)
        if video_timestamp <= last_timestamp + self.frame_time_interval:
            return
        timestamps = torch.arange(last_timestamp + self.frame_time_interval, video_timestamp, self.frame_time_interval) # add compensation
        
        # 3. fetch frames in required timestamps
        clip, clip_timestamps, clip_idxs = get_smart_resized_clip(video_reader, resized_height, resized_width, timestamps, video_pts, video_pts_index_from=last_video_pts_index+1)
        
        # Added protection: if no new frames are returned, we have hit the end of the video track
        if len(clip_idxs) == 0:
            state['video_end'] = True
            return

        state['last_video_pts_index'] = clip_idxs[-1]
        state['last_timestamp'] = clip_timestamps[-1]

        # 4. organize to interleave frames
        interleave_clips, interleave_timestamps = [], []
        if not initialized:
            interleave_clips.append(clip[:self.initial_fps_frames])
            interleave_timestamps.append(clip_timestamps[:self.initial_fps_frames])
            clip = clip[self.initial_fps_frames:]
            clip_timestamps = clip_timestamps[self.initial_fps_frames:]
        if len(clip) > 0:
            interleave_clips.extend(list(clip.split(self.streaming_fps_frames)))
            interleave_timestamps.extend(list(clip_timestamps.split(self.streaming_fps_frames)))

        # 5. make conversation and send to model
        for clip, timestamps in zip(interleave_clips, interleave_timestamps):
            start_timestamp, stop_timestamp = timestamps[0].item(), timestamps[-1].item() + self.frame_time_interval
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": f'Time={start_timestamp:.1f}-{stop_timestamp:.1f}s'},
                    {"type": "video", "video": clip}
                ]
            }
            if not query and not state.get('query', None):
                query = default_query
                print(f'No query provided, use default_query={default_query}')
            if query and state.get('query', None) != query:
                message['content'].append({"type": "text", "text": query})
                state['query'] = query
            texts = self.processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True, return_tensors='pt')
            past_ids = state.get('past_ids', None)
            if past_ids is not None:
                texts = '<|im_end|>\n' + texts[self.system_prompt_offset:]
            inputs = self.processor(
                text=texts,
                images=None,
                videos=[clip],
                return_tensors="pt",
                return_attention_mask=True # Set to True to completely resolve the attention mask warning
            )
            inputs.to('cuda')
            if past_ids is not None:
                inputs['input_ids'] = torch.cat([past_ids, inputs.input_ids], dim=1) 
            outputs = self.model.generate(
                **inputs, past_key_values=state.get('past_key_values', None), 
                return_dict_in_generate=True, do_sample=do_sample, 
                repetition_penalty=repetition_penalty,
                pad_token_id=self.model.config.eos_token_id,
                temperature=None, top_p=None, top_k=None # Clear config fallback properties to silence do_sample mismatch warnings
            )
            state['past_key_values'] = outputs.past_key_values
            state['past_ids'] = outputs.sequences[:, :-1]
            yield (start_timestamp, stop_timestamp), self.processor.decode(outputs.sequences[0, inputs.input_ids.size(1):], skip_special_tokens=True), state


# Main code : 
import cv2  # Required for displaying the image and saving the video
import torch
from tqdm import tqdm  # Required for the terminal progress bar

def overlay_text_bottom_left(img, text):
    """Helper function to burn text aligned to the bottom-left corner with a dark background box."""
    if not text:
        return img
    img_h, img_w, _ = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    font_color = (255, 255, 255)  # White
    thickness = 1
    line_type = cv2.LINE_AA
    
    text_position = (15, img_h - 25)
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    cv2.rectangle(
        img, 
        (10, img_h - 25 - text_size[1] - 5), 
        (15 + text_size[0] + 5, img_h - 15), 
        (0, 0, 0), 
        -1
    )
    cv2.putText(img, text, text_position, font, font_scale, font_color, thickness, line_type)
    return img

model_path = 'chenjoya/LiveCC-7B-Instruct'
video_path = "my-media/cats.mp4"
query = "Please describe the video."

infer = LiveCCDemoInfer(model_path=model_path)
state = {'video_path': video_path}
commentaries = []
t = 0

# A constant window name to ensure frames replace each other in a single window during Step 1
LIVE_WINDOW_NAME = "LiveCC Model Processing Stream"

# --- STEP 1: Generate all commentaries and display the first frame of each window ---
print("Generating commentaries for the video...")
for t in range(31):
    state['video_timestamp'] = t
    for (start_t, stop_t), response, state in infer.live_cc(
        query=query, state=state, 
        max_pixels = 384 * 28 * 28, repetition_penalty=1.05, 
    ):
        print(f'Generated ({start_t}s-{stop_t}s): {response}')
        clean_response = response.replace('\n', ' ')
        commentaries.append({
            'start': start_t,
            'stop': stop_t,
            'text': clean_response
        })
        
        # Extract the video reader and timestamps
        video_reader, _, _ = infer._cached_video_readers_with_hw[video_path]
        video_pts = state['video_pts']
        
        # Locate the exact frame index corresponding to the start of this timeframe (start_t)
        live_frame_idx = torch.argmin(torch.abs(video_pts - start_t)).item()
        live_frame_rgb = video_reader.get_batch([live_frame_idx]).asnumpy()[0]
        live_frame_bgr = cv2.cvtColor(live_frame_rgb, cv2.COLOR_RGB2BGR)
        
        # 1. Overlay the description text to the bottom-left corner
        live_frame_bgr = overlay_text_bottom_left(live_frame_bgr, clean_response)
        
        # 2. Format and overlay the starting time string (e.g. "0.0s") on the top-right corner
        time_text = f"{start_t:.1f}s"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (0, 255, 255)  # Cyan
        thickness = 1
        text_size, _ = cv2.getTextSize(time_text, font, font_scale, thickness)
        
        height, width, _ = live_frame_bgr.shape
        top_right_pos = (width - text_size[0] - 15, 25)
        
        # Draw background rectangle for the timestamp
        cv2.rectangle(
            live_frame_bgr, 
            (width - text_size[0] - 20, 10), 
            (width - 10, 32), 
            (0, 0, 0), 
            -1
        )
        cv2.putText(live_frame_bgr, time_text, top_right_pos, font, font_scale, font_color, thickness, cv2.LINE_AA)
        
        # Render the first frame of the current window to the active GUI canvas
        cv2.imshow(LIVE_WINDOW_NAME, live_frame_bgr)
        cv2.waitKey(1)  # Refresh window canvas
        
    if state.get('video_end', False):
        break
    t += 1

cv2.destroyAllWindows()

# --- STEP 2: Compile and Save all frames sequentially (No GUI Playback) ---
print("\n[LOG] Initializing video rendering process...")
video_reader, _, _ = infer._cached_video_readers_with_hw[video_path]
video_pts = state['video_pts'].tolist()  
total_frames = len(video_pts)

frame_rgb_sample = video_reader.get_batch([0]).asnumpy()[0]
height, width, _ = frame_rgb_sample.shape

src_cap = cv2.VideoCapture(video_path)
source_fps = src_cap.get(cv2.CAP_PROP_FPS)
src_cap.release()

if source_fps <= 0:
    source_fps = 25.0
    print(f"[WARNING] Could not parse source FPS automatically. Falling back to default: {source_fps}")
else:
    print(f"[LOG] Detected source video properties: Resolution={width}x{height}, FPS={source_fps}")

output_path = "my-media/output_commentary_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, source_fps, (width, height))

print(f"[LOG] Processing a total of {total_frames} frames into output video structure. Please wait...")

for frame_idx in tqdm(range(total_frames), desc="Compiling Final Video File"):
    current_time = video_pts[frame_idx]
    
    active_text = ""
    for segment in commentaries:
        if segment['start'] <= current_time < segment['stop']:
            active_text = segment['text']
            break
            
    frame_rgb = video_reader.get_batch([frame_idx]).asnumpy()[0]
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    # Burn the bottom-left description box only. Top-right timestamp is skipped to keep output file clean
    frame_bgr = overlay_text_bottom_left(frame_bgr, active_text)
    video_writer.write(frame_bgr)

video_writer.release()
print(f"\n[LOG] Finished compiling output video successfully! File saved to: {output_path}")