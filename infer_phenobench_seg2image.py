from share import *
import config

import os
import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
from tqdm import tqdm
from glob import glob
from random import choice

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
# from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


# apply_uniformer = UniformerDetector()

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('/netscratch/naeem/controlnet/phenobench_train/lightning_logs/version_1252437/checkpoints/epoch42-step=60500_backup.ckpt', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(input_image, 
            prompt, 
            a_prompt="best quality, extremely detailed, realistic, agricultural", 
            n_prompt="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, cartoonish, low quality, deformed, ugly, blurry, low resolution, low quality", 
            num_samples=1, 
            # image_resolution, 
            # detect_resolution, 
            ddim_steps=20, 
            guess_mode=False, 
            strength=0.6, 
            scale=8, 
            seed=10, 
            # eta
            ):
    with torch.no_grad():
        # H, W, C = img.shape
        input_image = cv2.imread(input_image)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        H, W, C = 1024, 1024, 3
        resized_image = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_NEAREST)

        control = torch.from_numpy(resized_image.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, 
                                                    #  eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        # results = x_samples[0] for i in range(num_samples)
    return x_samples[0]

# input_mask = cv2.imread("/netscratch/naeem/phenobench/plants_panoptic_val/05-15_00180_P0030686_panoptic.png")
input_folder = '/netscratch/naeem/sugarbeet_syn_v6/plants_panoptic_train/'
target_dir = '/netscratch/naeem/sugarbeet_syn_v6/controlnet_images/'
os.makedirs(target_dir, exist_ok=True)
images = sorted(glob(os.path.join(input_folder, '*.png')))
prompts = [
        "sugarbeet crops and weed plants of different species with dark green colored leaves from early growth stages with sunny lighting conditions in the morning and dry darker brown soil background",
        "sugarbeet crops and weed plants of different species with dark green colored leaves from early stages with sunny lighting conditions in the afternoon and dry lighter brown soil background",
        "sugarbeet crops and weed plants of different species with dark green colored leaves from later growth stages with overcast weather conditions without shadows and dark brown soil background with a bit of moisture"
        ]

for idx, input_mask in tqdm(enumerate(images)):
    result = process(input_image=input_mask, prompt=choice(prompts))
    base_name = os.path.basename(input_mask)
    cv2.imwrite(os.path.join(target_dir, base_name[:4]+'.png'), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))