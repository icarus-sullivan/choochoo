# choochoo

High-performance LoRA fine-tuning for video and image diffusion models.

Supports **WAN 2.2** (text-to-video, image-to-video), **Qwen-Image** (text-to-image), **Qwen Image Edit** (instruction-based editing), and **LTX-2** (stub, coming soon).

Features: zero-config AutoTuner, dual-LoRA single-pass training, FSDP/DDP multi-GPU, convergence detection, named safetensors output.

---

## Install

**Requirements:** Python 3.10+, PyTorch 2.2+, CUDA 12.1+

```bash
git clone https://github.com/icarus-sullivan/choochoo
cd choochoo
pip install -e .
```

Flash Attention and xFormers are optional but recommended on Linux:

```bash
pip install flash-attn xformers
```

To inspect training metrics from the SQLite DB written to `output_dir` (optional):

```bash
# Linux / WSL
apt install sqlite3

# macOS
brew install sqlite

# or use any SQLite GUI (DB Browser, TablePlus, etc.)
sqlite3 output/qwen-myrun.db "SELECT step, loss, lr, phase FROM metrics ORDER BY step;"
```

---

## Prepare your dataset

Training requires real data. choochoo will not start without a valid `data_dir`.

### WAN 2.2 (video)

Place video files in a directory. Add a caption per video as a `.txt` file with the same stem, or use a single `metadata.jsonl`:

```
my-videos/
  clip_001.mp4
  clip_001.txt          ← "a dog running through a field"
  clip_002.mp4
  clip_002.txt
```

Or with `metadata.jsonl` (takes precedence over `.txt` files):

```jsonl
{"file_name": "clip_001.mp4", "caption": "a dog running through a field"}
{"file_name": "clip_002.mp4", "caption": "a cat sitting on a windowsill"}
```

Supported formats: `.mp4` `.mov` `.avi` `.mkv` `.webm`

### Qwen-Image

Same format as WAN — images with captions:

```
my-images/
  photo_001.png
  photo_001.txt     ← "a golden retriever running on the beach"
  photo_002.png
  photo_002.txt
```

Or a `metadata.jsonl`: `{"file_name": "photo_001.png", "caption": "..."}`

No source image needed — pure text-to-image training.

### Qwen Image Edit

Same format — images with captions describing the desired edit style or instruction:

```
my-edits/
  image_001.png
  image_001.txt     ← "a portrait with dramatic lighting"
  image_002.png
  image_002.txt
```

Or a `metadata.jsonl`: `{"file_name": "image_001.png", "caption": "..."}`

---

## Quick start

Copy a minimal example config and fill in your paths:

```bash
cp examples/wan22_minimal.yaml my_run.yaml
```

Edit `my_run.yaml`:

```yaml
name: my_concept

model:
  type: wan22
  pretrained_path: /path/to/wan22-model

data:
  data_dir: /path/to/videos

lora:
  rank: 16
  alpha: 32

training:
  max_steps: 2000
  learning_rate: 1.0e-4
  save_every: 500
```

Then train:

```bash
python train.py --config my_run.yaml
```

Checkpoints and LoRA weights are written to `./output/` by default.

---

## Configs

| File | Purpose |
|------|---------|
| `examples/wan22_minimal.yaml` | Minimal WAN 2.2 config — required fields only |
| `examples/wan22_lora.yaml` | WAN 2.2 with common options set |
| `examples/qwen_minimal.yaml` | Minimal Qwen-Image config (pure T2I) |
| `examples/qwen_edit_minimal.yaml` | Minimal Qwen Image Edit config |
| `examples/qwen_edit_lora.yaml` | Qwen Image Edit with common options set |
| `examples/full_options.yaml` | Every supported option, documented |

Any key not present in your config falls back to its default automatically.

---

## Multi-GPU

Single node, 4 GPUs:

```bash
torchrun --nproc_per_node=4 train.py --config my_run.yaml
```

Multi-node (2 nodes, 8 GPUs each):

```bash
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
  --master_addr=<host> --master_port=29500 \
  train.py --config my_run.yaml
```

The distributed strategy (`ddp` / `fsdp`) is auto-selected based on GPU count. Override with `distributed.strategy` in your config.

---

## Output files

LoRA weights are saved as safetensors using the run `name` from your config:

| Model | Output files |
|-------|-------------|
| WAN 2.2 (single LoRA) | `wan-<name>-step<N>.safetensors` |
| WAN 2.2 (dual LoRA) | `wan-<name>-high-step<N>.safetensors` + `wan-<name>-low-step<N>.safetensors` |
| Qwen-Image | `qwen-<name>-step<N>.safetensors` |
| Qwen Image Edit | `qwen-<name>-step<N>.safetensors` |

---

## CLI flags

```
python train.py --config <path>          # required
                --resume <checkpoint>    # resume from checkpoint directory
                --pretrained-path <path> # override model.pretrained_path
                --data-dir <path>        # override data.data_dir
                --output-dir <path>      # override logging.output_dir
                --max-steps <N>          # override training.max_steps
                --no-auto-tune           # skip AutoTuner, use config values directly
                --merge-lora <ckpt>      # merge LoRA into base model and exit
                --export-lora <ckpt>     # export LoRA weights and exit
```

---

## Resume training

```bash
python train.py --config my_run.yaml --resume ./output/checkpoint-1000
```
