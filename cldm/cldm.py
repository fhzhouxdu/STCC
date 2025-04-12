
import einops
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import glob
import os
import cv2
import numpy as np
import math
from timm.models.vision_transformer import Block

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)


from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer, PixelAwareTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from .uncertainty import UncertaintyWeigh
import sys


class ControlledUnetModel(UNetModel):
    def __init__(
        self, 
        image_size, 
        in_channels, 
        model_channels, 
        out_channels, 
        num_res_blocks, 
        attention_resolutions, 
        dropout=0, 
        channel_mult=..., 
        conv_resample=True, 
        dims=2, 
        num_classes=None, 
        use_checkpoint=False, 
        use_fp16=False, 
        num_heads=-1, 
        num_head_channels=-1, 
        num_heads_upsample=-1, 
        use_scale_shift_norm=False, 
        resblock_updown=False, 
        use_new_attention_order=False, 
        use_spatial_transformer=False, 
        transformer_depth=1, 
        context_dim=None, 
        n_embed=None, 
        legacy=True, 
        disable_self_attentions=None, 
        num_attention_blocks=None, 
        disable_middle_self_attn=False, 
        use_linear_in_transformer=False,
        ):
        super().__init__(
        image_size, 
        in_channels, 
        model_channels, 
        out_channels, 
        num_res_blocks, 
        attention_resolutions, 
        dropout, 
        channel_mult, 
        conv_resample, 
        dims, 
        num_classes, 
        use_checkpoint, 
        use_fp16, 
        num_heads, 
        num_head_channels, 
        num_heads_upsample, 
        use_scale_shift_norm, 
        resblock_updown, 
        use_new_attention_order, 
        use_spatial_transformer, 
        transformer_depth, 
        context_dim, 
        n_embed, 
        legacy, 
        disable_self_attentions, 
        num_attention_blocks, 
        disable_middle_self_attn, 
        use_linear_in_transformer
        )
            
    def forward(self, x, timesteps=None, context=None, control=None, Y=None, only_mid_control=False,scale=None,mode=None,impath=None,**kwargs):
        hs = []
        cs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h= module(h, emb, context)
                hs.append(h)
                
            # cross normalization
            if control is not None:
                controls = control[-1]                
                mean_latents, std_latents = torch.mean(h, dim=(1, 2, 3), keepdim=True), torch.std(h, dim=(1, 2, 3), keepdim=True)
                mean_control, std_control = torch.mean(controls, dim=(1, 2, 3), keepdim=True), torch.std(controls, dim=(1, 2, 3), keepdim=True)
                controls = (controls - mean_control) * (std_latents / (std_control + 1e-12)) + mean_latents
                h = h+controls
                
            h = self.middle_block(h, emb,context)
        
  
        if control is not None:
            h += control.pop()
        
        for i, module in enumerate(self.output_blocks):
            
            h = torch.cat([h, hs.pop()+control.pop()], dim=1)
            h = module(h, emb, context, Y.pop())
                
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

        
    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb)

        outs = []
        y = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            y.append(h)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs,y


def dis_reg(scale,cross_init):
    b,c,h,w =scale.shape
    scale[cross_init<0.5] = 0.1
    return scale

class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, global_average_pooling=False, stage="train", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self.global_average_pooling = global_average_pooling
        self.stage = stage
        
        self.shared_encoder = nn.Sequential(*[
            Block(
                dim=1024,
                num_heads=8,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            )
            for i in range(8)]).to('cuda')
        
        # add missing_prompt
        prompt_num = 1
        prompt_length = 16
        embed_dim = 1024
        complete_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        complete_prompt[:,0:1,:].fill_(1)
        complete_prompt[:,prompt_length//2:prompt_length//2+1,:].fill_(1)
        self.complete_prompt = nn.Parameter(complete_prompt)
        missing_text_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        missing_text_prompt[:,1:2,:].fill_(1)
        missing_text_prompt[:,prompt_length//2+1:prompt_length//2+2,:].fill_(1)
        self.missing_text_prompt = nn.Parameter(missing_text_prompt)
        self.feats_attn = Attention(dim=1024)
        
        # weight predict
        self.uncertainty_weigh = UncertaintyWeigh(2)
        self.uncertainty_loss = None



    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        clip = batch['clip']
        emp = batch['emp']
        emp = self.get_learned_conditioning(emp)
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        clip = einops.rearrange(clip, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        clip = clip.to(memory_format=torch.contiguous_format).float()
        clip = clip.to(self.device)
        return x, dict(c_crossattn=[c], c_concat=[control], clip=[clip], emp=[emp])

    def apply_model(self, x_noisy, t, cond, mode='c_ada', impath=None, *args, **kwargs):
        assert isinstance(cond, dict)
        import torch.nn.functional as F

        diffusion_model = self.model.diffusion_model

        cond_emp = torch.cat(cond['emp'], 1)
        
        cond_txt = torch.cat(cond['c_crossattn'], 1)
        cond_img = self.cond_stage_model.model.encode_image(torch.cat(cond['clip'], 1)).unsqueeze(1)
        cond_img = self.shared_encoder(cond_img)
        cond_img = F.normalize(cond_img, dim=1)
        if self.stage == "train":
            cond_txt = self.shared_encoder(cond_txt)
            cond_txt = F.normalize(cond_txt, dim=1)
            prompt = self.complete_prompt
            cond_cat,_ = self.feats_attn(torch.cat((cond_txt, cond_img), dim=1), prompts=prompt)
            cond_txt, cond_img = cond_cat[:,:-1,:], cond_cat[:,-1:,:]
            w_txt, w_img = self.weight_predictor(cond_txt, cond_img)
            loss_uncertainty, w = self.uncertainty_weigh(cond_txt, cond_img)
            self.uncertainty_loss = loss_uncertainty
            cond_fus = w[:,0,:]*cond_txt + w[:,1,:]*cond_img
            
        else:
            cond_txt = cond_emp
            cond_txt = self.shared_encoder(cond_txt)
            cond_txt = F.normalize(cond_txt, dim=1)
            prompt = self.missing_text_prompt
            cond_cat,_ = self.feats_attn(torch.cat((cond_txt, cond_img), dim=1), prompts=prompt)
            cond_txt, cond_img = cond_cat[:,:-1,:], cond_cat[:,-1:,:]
            _, w = self.uncertainty_weigh(cond_txt, cond_img)
            cond_fus = w[:,0,:]*cond_txt + w[:,1,:]*cond_img


        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_emp, control=None, only_mid_control=self.only_mid_control)
        else:
            control,y = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_fus)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, Y=y, only_mid_control=self.only_mid_control)
        
        return eps

    
    def p_losses(self, x_start, cond, t, noise=None):
        loss, loss_dict = super().p_losses(x_start, cond, t, noise)
        if self.stage == "train":
            loss += 0.01*self.uncertainty_loss
            prefix = 'train' if self.training else 'val'
            loss_dict.update({f'{prefix}/loss_unceratinty': 0.01*self.uncertainty_loss})
        return loss, loss_dict

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c, clip, emp = c["c_concat"][0][:N], c["c_crossattn"][0][:N], c["clip"][0][:N], c["emp"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross], "clip": [clip], "emp": [emp]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c], "clip": [clip], "emp":[emp]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond,mode='c_ada', verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        pix_params = []
        for module in self.model.diffusion_model.output_blocks.named_modules():
            if isinstance(module[1], PixelAwareTransformer):
                pix_params += module[1].parameters()
        params = list(self.control_model.parameters())
        params += pix_params
        params += list(self.shared_encoder.parameters())
        params += list(self.feats_attn.parameters())
        if self.stage == "train":
            params += [self.complete_prompt]
            params += list(self.uncertainty_weigh.parameters())
        elif self.stage == "finetune":
            params += [self.missing_text_prompt]
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None, prompts=None, learnt_p=True):
        B, N, C = x.shape
        prompts = prompts.repeat(B,1,1)
        # prefix prompt tuning
        P = prompts.size(1)
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, C)
        )   
        
        if learnt_p:
            P = P//2
            prompts_k = prompts[:,:P]
            prompts_v = prompts[:,P:]
        else:
            prompts_k = prompts
            prompts_v = prompts

        q, k, v = (
            qkv[:,:,0,:].reshape(B,N,self.num_heads,C//self.num_heads).permute(0,2,1,3),
            torch.cat([prompts_k,qkv[:,:,1,:]], dim=1).reshape(B,N+P,self.num_heads,C//self.num_heads).permute(0,2,1,3),
            torch.cat([prompts_v,qkv[:,:,2,:]], dim=1).reshape(B,N+P,self.num_heads,C//self.num_heads).permute(0,2,1,3),
        )  

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn
    