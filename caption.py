import os
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse

# ========================= CONFIG =========================
MODEL_ID = "prithivMLmods/Qwen3-VL-8B-Abliterated-Caption-it"

SYSTEM_PROMPT = (
    "You are a precise, exhaustive visual captioner generating training data for image models. "
    "Describe every visible element in full detail: subjects, anatomy, clothing and its state, "
    "poses, expressions, interactions, setting, lighting, color, texture, and composition. "
    "Be completely literal and objective. Output only the caption as a single detailed paragraph."
)
# =========================================================

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", type=str, required=True, help="Folder containing images and videos")
parser.add_argument("--trigger", type=str, required=True, help="LoRA trigger word prepended to each caption")
parser.add_argument("--batch-size", type=int, default=24, help="Number of images to process simultaneously")
parser.add_argument("--skip-existing", action="store_true", help="Skip files that already have .txt")
parser.add_argument("--max-frames", type=int, default=32, help="Max frames to sample from video (higher = better but slower)")
args = parser.parse_args()

print("Loading Qwen3-VL model (this may take a while the first time)...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

SUPPORTED = ('.png', '.jpg', '.jpeg', '.webp', '.avif', '.mp4', '.m4v', '.avi', '.mov', '.webm')

def build_messages(media_path: str) -> list:
    is_img = media_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
    if is_img:
        vision_item = {"type": "image", "image": os.path.abspath(media_path)}
    else:
        vision_item = {"type": "video", "video": media_path, "max_frames": args.max_frames}

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            vision_item,
            {"type": "text", "text": "Describe this in maximum explicit detail for LoRA training."}
        ]}
    ]

def caption_batch(media_paths: list) -> list:
    try:
        all_texts = []
        all_image_inputs = []
        all_video_inputs = []

        for path in media_paths:
            messages = build_messages(path)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            all_texts.append(text)
            if image_inputs:
                all_image_inputs.extend(image_inputs)
            if video_inputs:
                all_video_inputs.extend(video_inputs)

        inputs = processor(
            text=all_texts,
            images=all_image_inputs or None,
            videos=all_video_inputs or None,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        captions = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return [f"{args.trigger} {c.strip()}" for c in captions]

    except Exception as e:
        return [f"ERROR: {str(e)[:200]}"] * len(media_paths)

# ====================== Batch Processing ======================
pending = []  # list of (media_path, txt_path)

for root, _, files in os.walk(args.input_dir):
    for file in sorted(files):
        if not file.lower().endswith(SUPPORTED):
            continue
        media_path = os.path.join(root, file)
        if file.lower().endswith('.avif'):
            png_path = os.path.splitext(media_path)[0] + '.png'
            Image.open(media_path).convert('RGB').save(png_path)
            os.remove(media_path)
            media_path = png_path
            file = os.path.basename(png_path)
        txt_path = os.path.splitext(media_path)[0] + ".txt"
        if args.skip_existing and os.path.exists(txt_path):
            print(f"Skipping {file} (caption exists)")
            continue
        pending.append((media_path, txt_path))

for i in range(0, len(pending), args.batch_size):
    batch = pending[i:i + args.batch_size]
    paths = [p for p, _ in batch]
    print(f"Captioning batch: {[os.path.basename(p) for p in paths]}")

    captions = caption_batch(paths)

    for (media_path, txt_path), caption in zip(batch, captions):
        if caption.startswith("ERROR:"):
            print(f"  Error on {os.path.basename(media_path)}: {caption}")
        else:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption)
            print(f"  → Saved: {txt_path}")

print("Batch captioning finished!")
