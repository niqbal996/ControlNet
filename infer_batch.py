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
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize

class ImageDataset(Dataset):
    def __init__(self, image_paths, target_size=(1024, 1024)):
        self.image_paths = image_paths
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_NEAREST)
        img = torch.from_numpy(img).float() / 255.0
        img = einops.rearrange(img, 'h w c -> c h w')
        return img, os.path.basename(img_path)

def process_batch(model, ddim_sampler, batch, prompts, config,
                  a_prompt="best quality, extremely detailed, realistic, agricultural",
                  n_prompt="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, cartoonish, low quality, deformed, ugly, blurry, low resolution, low quality",
                  ddim_steps=20,
                  guess_mode=False,
                  strength=0.6,
                  scale=8,
                  seed=10):
    
    with torch.no_grad():
        batch_size = batch.shape[0]
        control = batch.cuda()
        
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt for prompt in prompts])]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * batch_size)]}
        
        shape = (4, control.shape[2] // 8, control.shape[3] // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)

        samples, _ = ddim_sampler.sample(ddim_steps, batch_size,
                                         shape, cond, verbose=False,
                                         unconditional_guidance_scale=scale,
                                         unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

    return x_samples

def main():
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict('/netscratch/naeem/controlnet/phenobench_train/lightning_logs/version_1252437/checkpoints/epoch42-step=60500_backup.ckpt', location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    input_folder = '/netscratch/naeem/sugarbeet_syn_v6/plants_panoptic_train/'
    target_dir = '/netscratch/naeem/sugarbeet_syn_v6/controlnet_images/'
    os.makedirs(target_dir, exist_ok=True)

    images = sorted(glob(os.path.join(input_folder, '*.png')))[480:]
    dataset = ImageDataset(images)
    batch_size = 3
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    prompts = [
        "sugarbeet crops and weed plants of different species with dark green colored leaves from early growth stages with sunny lighting conditions in the morning and dry darker brown soil background",
        "sugarbeet crops and weed plants of different species with dark green colored leaves from early stages with sunny lighting conditions in the afternoon and dry lighter brown soil background",
        "sugarbeet crops and weed plants of different species with dark green colored leaves from later growth stages with overcast weather conditions without shadows and dark brown soil background with a bit of moisture"
    ]

    for batch, filenames in tqdm(dataloader):
        batch_prompts = [random.choice(prompts) for _ in range(len(batch))]
        results = process_batch(model, ddim_sampler, batch, batch_prompts, config)
        
        for result, filename in zip(results, filenames):
            cv2.imwrite(os.path.join(target_dir, filename[:4]+'.png'), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    main()
