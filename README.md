Qwen-Image Text-to-Image Generation with Low Memory Requirements
===

# Requirements
- NVIDIA GPU with at least 16GB RAM (e.g. 4070)
- Python 3.10
- Basic knowledge of Python programming
- Git installed

# Installation
1. Clone this repository: `git clone https://github.com/shivanraptor/qwen-image-t2i.git`
2. Create a virtual environment, preferably using Python 3.10 as the environment: `python -m venv t2ivenv`
3. Activate your virtual environment: `. t2ivenv/bin/activate`
4. Install the required packages via pip: `pip install -r requirements.txt`
5. (Optional) Download your favorite LoRA and put it in the `lora/` folder
6. Edit the `config.cfg` for Prompt and Aspect Ratio (see config parameters details below), optionally set the LoRA
7. Execute the command: `python qwen.py config.cfg`
8. You will find the generated image in the `output/` folder
9. Deactivate the virtual environment afterwards: `deactivate`

Enjoy!

# Configurable Parameters

Sample config file content:
```
prompt = "A Japanese young lady with long black hair, 20-year-old, wearing school uniform, casually leaning to a traffic light, in an urban Hong Kong setting. Her pose and expression convey confidence and comfort. photorealistic, Ultra HD, 4K, cinematic composition"
aspect_ratio = "9:16"
lora = ""
```

`prompt`: Describe your image as detailed as possible.

`aspect_ratio`: There are several preset aspect ratios available: '1:1', '16:9', '9:16', '4:3', '3:4', '3:2', '2:3'

| aspect_ratio | width | height |
| ----------- | ----------- | ----------- |
| 1:1 | 1328 | 1328 |
| 16:9 | 1664 | 928 |
| 9:16 | 928 | 1664 |
| 4:3 | 1472 | 1104 |
| 3:4 | 1104 | 1472 | 
| 3:2 | 1584 | 1056 |
| 2:3 | 1056 | 1584 |

`lora`: the LoRA name you downloaded and saved in the `lora/` folder (no space in the filename); if you don't have a LoRA, leave it blank.

# Support
Use the Discussions for questions, or Issues for bugs in this GitHub repository. I will get back to you shortly.
