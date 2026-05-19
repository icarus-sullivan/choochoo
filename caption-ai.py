import os
import subprocess
import torch
import ffmpeg
import numpy as np
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import argparse

# ========================= CONFIG =========================
MODEL_ID = "fancyfeast/llama-joycaption-beta-one-hf-llava"

SYSTEM_PROMPT = (
    "You are a precise, exhaustive captioner generating explicit training data "
    "for adult image models. Describe every visible element with full anatomical "
    "detail: subjects, body parts, exact physical acts, positions, penetration, "
    "clothing state, expressions, setting, lighting, camera angle, and composition. "
    "Be completely literal and clinical. Output only the caption."
    "Ignore all overlays or non-people based details"
)

USER_PROMPT = (
    "Write a short, detailed, explicit caption describing everything happening "
    "in this image for LoRA training data. Include all anatomical and sexual details."
    "Write it in a natural language way as if one guy telling another guy."
    "Be very terse only describing actions the person is doing in a sexual way."
    "Call out sex positions, fluids and facial expressions."
)
# =========================================================

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", type=str, required=True)
parser.add_argument("--trigger", type=str, required=True)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--skip-existing", action="store_true")
parser.add_argument("--max-frames", type=int, default=8)
parser.add_argument("--frame-tokens", type=int, default=180, help="Max tokens per video frame caption")
args = parser.parse_args()

print("Loading JoyCaption Beta One model...")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype="bfloat16",
    device_map=0
)
model.eval()

SUPPORTED = ('.png', '.jpg', '.jpeg', '.webp', '.avif',
             '.mp4', '.m4v', '.avi', '.mov', '.webm')
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.webp')

# ===================== FFMPEG =====================

def ffmpeg_extract_frames(video_path, max_frames):
    probe = ffmpeg.probe(video_path)
    stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    width = int(stream["width"])
    height = int(stream["height"])
    duration = float(stream.get("duration", 0))
    sample_fps = max_frames / duration if duration > 0 else 1

    out, _ = (
        ffmpeg.input(video_path)
        .filter("fps", fps=sample_fps)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run(capture_stdout=True, capture_stderr=True)
    )

    video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
    return [Image.fromarray(frame) for frame in video]

# ===================== CAPTION HELPERS =====================

def make_convo_string():
    convo = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    convo_string = processor.apply_chat_template(
        convo, tokenize=False, add_generation_prompt=True
    )
    assert isinstance(convo_string, str), f"Expected str, got {type(convo_string)}"
    return convo_string

CONVO_STRING = None  # cached after model load

def get_convo_string():
    global CONVO_STRING
    if CONVO_STRING is None:
        CONVO_STRING = make_convo_string()
    return CONVO_STRING

# ===================== SINGLE CAPTION =====================

def run_caption(image, max_new_tokens=512):
    convo_string = get_convo_string()

    inputs = processor(
        text=[convo_string],
        images=[image],
        return_tensors="pt"
    ).to(model.device)
    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

    input_len = inputs['input_ids'].shape[1]

    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            suppress_tokens=None,
            use_cache=True,
            temperature=0.6,
            top_k=None,
            top_p=0.9,
            pad_token_id=processor.tokenizer.eos_token_id,
        )[0]

    generate_ids = generate_ids[input_len:]
    return processor.tokenizer.decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()

# ===================== BATCH IMAGES =====================

def run_caption_batch_images(images):
    convo_string = get_convo_string()
    texts = [convo_string] * len(images)

    try:
        inputs = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

        input_len = inputs['input_ids'].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                suppress_tokens=None,
                use_cache=True,
                temperature=0.6,
                top_k=None,
                top_p=0.9,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

        captions = []
        for out in outputs:
            gen = out[input_len:]
            captions.append(
                processor.tokenizer.decode(
                    gen, skip_special_tokens=True, clean_up_tokenization_spaces=False
                ).strip()
            )
        return captions

    except Exception:
        return [run_caption(img) for img in images]

# ===================== BATCH DISPATCH =====================

def caption_batch(paths):
    img_indices = []
    img_paths = []
    results = {}

    for i, path in enumerate(paths):
        if path.lower().endswith(IMG_EXTS):
            img_indices.append(i)
            img_paths.append(path)
        else:
            try:
                frames = ffmpeg_extract_frames(path, args.max_frames)
                frame_caps = [run_caption(f, max_new_tokens=args.frame_tokens) for f in frames]
                results[i] = " ".join(frame_caps)
            except Exception as e:
                results[i] = f"ERROR: {str(e)[:200]}"

    if img_paths:
        try:
            images = [Image.open(p).convert("RGB") for p in img_paths]
            captions = run_caption_batch_images(images)
            for idx, cap in zip(img_indices, captions):
                results[idx] = cap
        except Exception:
            for idx, p in zip(img_indices, img_paths):
                try:
                    results[idx] = run_caption(Image.open(p).convert("RGB"))
                except Exception as e2:
                    results[idx] = f"ERROR: {str(e2)[:200]}"

    out = []
    for i in range(len(paths)):
        cap = results.get(i, "ERROR: missing")
        if not cap.startswith("ERROR"):
            cap = f"{args.trigger}, {cap}"
        out.append(cap)
    return out

# ===================== FILE SCAN =====================

def convert_avif(avif_path):
    png_path = os.path.splitext(avif_path)[0] + ".png"
    try:
        Image.open(avif_path).convert("RGB").save(png_path)
    except Exception:
        subprocess.run(
            ["ffmpeg", "-y", "-i", avif_path, png_path],
            check=True, capture_output=True
        )
    os.remove(avif_path)
    return png_path

pending = []

for root, _, files in os.walk(args.input_dir):
    for file in sorted(files):
        if not file.lower().endswith(SUPPORTED):
            continue

        path = os.path.join(root, file)

        if file.lower().endswith(".avif"):
            path = convert_avif(path)
            file = os.path.basename(path)

        txt_path = os.path.splitext(path)[0] + ".txt"

        if args.skip_existing and os.path.exists(txt_path):
            print(f"Skipping {file} (caption exists)")
            continue

        pending.append((path, txt_path))

# ===================== RUN =====================

for i in range(0, len(pending), args.batch_size):
    batch = pending[i:i + args.batch_size]
    paths = [p for p, _ in batch]

    print(f"Captioning batch: {[os.path.basename(p) for p in paths]}")

    captions = caption_batch(paths)

    for (path, txt_path), cap in zip(batch, captions):
        if cap.startswith("ERROR"):
            print(f"  Error on {os.path.basename(path)}: {cap}")
            continue
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(cap)
        print(f"  -> Saved: {txt_path}")

print("Done.")
