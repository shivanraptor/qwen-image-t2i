import os
# Optimize PyTorch memory allocation
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
# Enable memory-efficient attention if available
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Suppress Warnings (if any)
import logging
logging.getLogger().setLevel(logging.ERROR)

from diffusers import AutoModel, DiffusionPipeline
import torch
import random
import sys

### Helper functions
def load_config(config_path: str) -> dict:
    """Load and parse the TOML config file, using tomllib or toml based on Python version."""
    # Check Python version
    if sys.version_info >= (3, 11):
        import tomllib
        try:
            with open(config_path, 'rb') as file:
                config = tomllib.load(file)
            return config
        except tomllib.TOMLDecodeError as e:
            print(f"Error: Invalid TOML in '{config_path}': {e}")
            sys.exit(1)
    else:
        try:
            import toml
        except ImportError:
            print("Error: 'toml' library not installed. Install it with: pip install toml")
            sys.exit(1)
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = toml.load(file)
            return config
        except toml.TomlDecodeError as e:
            print(f"Error: Invalid TOML in '{config_path}': {e}")
            sys.exit(1)



## Phase 1: Fetch and Check Parameters
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("config", help="config file path (in TOML format)")
args = parser.parse_args()

# Load parameters from TOML Config File
config = None
if args.config is not None:
	config = load_config(args.config)
else:
	# should not fall into this. ArgumentParser handles the required parameter check
	print("Config file parameter is missing from the command.")
	sys.exit(1)

## Access values with automatic type conversion
prompt = config['prompt'] # str
aspect_ratio = config['aspect_ratio'] # str (Options: '1:1', '16:9', '9:16', '4:3', '3:4', '3:2', '2:3')
lora = config['lora'] # str ( empty or a filename in lora/ folder)

# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

if aspect_ratio not in list(aspect_ratios.keys()):
    aspect_ratio = "9:16"

width, height = aspect_ratios[aspect_ratio]

### Settings: Prompt
if prompt == "":
	print("ERROR: Prompt cannot be empty. Exiting...")
	sys.exit(1)

# Settings (no need to change)
model_name = "Qwen/Qwen-Image"

# Load the pipeline
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

transformer = AutoModel.from_pretrained(
    "dimitribarbot/Qwen-Image-int8wo",
    torch_dtype=torch_dtype,
    use_safetensors=False # the quantized model does not have a safetensor version
)
pipe = DiffusionPipeline.from_pretrained(
    model_name,
    transformer=transformer,
    torch_dtype=torch_dtype
)
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

# Load LoRA weights
if lora:
    if os.path.exists('lora/' + lora):
        pipe.load_lora_weights('lora/' + lora, adapter_name="lora")
    else:
        logger.error('Specified LoRA file cannot be found. Ignoring...')

negative_prompt =  "low quality, blurry, deformed, ugly, bad anatomy, extra limbs, watermark, text, logo, artist name, no distortion, no warped text, no duplicate faces, text"

seed = random.randint(0, 2**32 - 1)

image = pipe(
    prompt = prompt,
    negative_prompt = negative_prompt,
    width = width,
    height = height,
    num_inference_steps = 50,
    true_cfg_scale = 5,
    generator = torch.Generator(device="cuda").manual_seed(seed)
)
image = image.images[0]

image.save(f"output/output_{seed}.png")
