from share import *
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tutorial_dataset import TestDataset
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler
import einops
from PIL import Image
import numpy as np
import open_clip
import time

# Configs
resume_path = '/root/autodl-tmp/checkpoints/epoch=0-step=39.ckpt'
batch_size = 1
sd_locked = True
only_mid_control = False
ddim_steps = 50

# load smart control model
# model = create_model('./models/cldm_v21.yaml').cpu()
# checkpoint = torch.load(resume_path, map_location='cpu')
# target_dict = {}
# checkpoint = torch.load(resume_path, map_location='cpu')
# pretrained_weights = checkpoint['state_dict']
# for k in pretrained_weights.keys():
#     target_dict[k] = pretrained_weights[k].clone()   
# model.load_state_dict(target_dict,strict=False)

#load traditional controlnet
model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))

model.sd_locked = sd_locked
model.only_mid_control = only_mid_control
model.to('cuda:0').eval()
ddim_sampler = DDIMSampler(model)

#load datasets
dataset = TestDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=False)
for item in dataloader:
    filename = item['filename'][0]
    batch = item['data']
    with torch.no_grad():
        # print(model.cond_stage_model.model.encode_text(open_clip.tokenize(batch['txt']).to('cuda:0')).shape)
        _, c = model.get_input(batch, model.first_stage_key, bs=batch_size)
        # c_cat, c, clip, emp= c["c_concat"][0][:batch_size], c["c_crossattn"][0][:batch_size], c["clip"][0][:batch_size], c["emp"][0][:batch_size]
        c_cat, c = c["c_concat"][0][:batch_size], c["c_crossattn"][0][:batch_size]
        uc_cross = model.get_unconditional_conditioning(batch_size)
        uc_cat = c_cat  # torch.zeros_like(c_cat)
        # uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross], "clip": [clip], "emp": [emp]}
        uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
        start_time = time.time()
        samples, _ = model.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                            batch_size=batch_size, ddim=True,
                                            ddim_steps=ddim_steps, eta=0.0,
                                            unconditional_guidance_scale=9.,
                                            unconditional_conditioning=uc_full,
                                            )
        # samples, _ = model.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c], "clip": [clip], "emp":[emp]},
        #                                     batch_size=batch_size, ddim=True,
        #                                     ddim_steps=ddim_steps, eta=0.0,
        #                                     unconditional_guidance_scale=9.,
        #                                     unconditional_conditioning=uc_full,
        #                                     )
        end_time = time.time()
        break
        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp(x_samples, -1., 1.).cpu()
        x_samples = (x_samples + 1.0) / 2.0
        x_samples = x_samples.squeeze(0)
        x_samples = x_samples.transpose(0, 1).transpose(1, 2).squeeze(-1)
        x_samples = x_samples.numpy()
        x_samples = (x_samples * 255).astype(np.uint8)
        print(x_samples.shape)
        Image.fromarray(x_samples).save('./result/'+filename)
        
elapsed_time = end_time - start_time  # 计算耗时（单位：秒）

print(f"耗时: {elapsed_time:.4f} 秒")