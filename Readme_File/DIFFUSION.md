* Diffusion
    * [preload_models_from_standard_weights](#preload_models_from_standard_weights)
    

    
## preload_models_from_standard_weights


```
>>> from kumar.diffusion.model_loader import preload_models_from_standard_weights

preload_models_from_standard_weights(
    ckpt_path= ckpt_path,      # put in the ckpt_path file 
    device= device
)
```
- run the model in ipynb file Like this 
- make sure three file are found in the data folder 
    1. ../data/tokenizer_vocab.json         --> [https://huggingface.co/benjamin-paine/stable-diffusion-v1-5/blob/main/tokenizer/vocab.json]
    2. ../data/tokenizer_merges.txt         --> [https://huggingface.co/benjamin-paine/stable-diffusion-v1-5/blob/main/tokenizer/merges.txt]
    3. ../data/v1-5-pruned-emaonly.ckpt     --> [https://huggingface.co/benjamin-paine/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt]

```
import model_loader
import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch

DEVICE = "cpu"

ALLOW_CUDA = False
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

tokenizer = CLIPTokenizer("../data/tokenizer_vocab.json", merges_file="../data/tokenizer_merges.txt")
model_file = "../data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

## TEXT TO IMAGE

# prompt = "A dog with sunglasses, wearing comfy hat, looking at camera, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
prompt = "A cat stretching on the floor, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
uncond_prompt = ""  # Also known as negative prompt
do_cfg = True
cfg_scale = 8  # min: 1, max: 14

## IMAGE TO IMAGE

input_image = None
# Comment to disable image to image
# image_path = "../image/Dog_Breeds.jpg"
# input_image = Image.open(image_path)
# Higher values means more noise will be added to the input image, so the result will further from the input image.
# Lower values means less noise is added to the input image, so output will be closer to the input image.
strength = 0.9

## SAMPLER

sampler = "ddpm"
num_inference_steps = 50
seed = 42

output_image = pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    input_image=input_image,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    seed=seed,
    models=models,
    device=DEVICE,
    idle_device="cpu",
    tokenizer=tokenizer,
)

# Combine the input image and the output image into a single image.
Image.fromarray(output_image)
```

