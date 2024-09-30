import os
import torch
import folder_paths
import comfy.model_management as mm

import argparse
from omegaconf import OmegaConf
import logging
from .sgm.util import instantiate_from_config
from .inference.sample_func import sample_video, decode_video

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

script_directory = os.path.dirname(os.path.abspath(__file__))

class LoadLVCDModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "Normal SVD model, default is the normal very first non-XT SVD"} ),
                "use_xformers": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "precision": (["fp16", "fp32", "bf16"],
                    {"default": "fp16"}
                ),
            }
        }

    RETURN_TYPES = ("LVCDPIPE",)
    RETURN_NAMES = ("LVCD_pipe", )
    FUNCTION = "loadmodel"
    CATEGORY = "ComfyUI-LVCDWrapper"

    def loadmodel(self, model, precision, use_xformers):
        
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        mm.soft_empty_cache()

        svd_model_path = folder_paths.get_full_path_or_raise("checkpoints", model)
        download_path = os.path.join(folder_paths.models_dir, "lvcd")
        lvcd_path = os.path.join(download_path, "lvcd-fp16.safetensors")

        if not os.path.exists(lvcd_path):
            log.info(f"Downloading LVCD model to: {lvcd_path}")
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id="Kijai/LVCD-pruned",
                local_dir=download_path,
                local_dir_use_symlinks=False,
            )

        config_path = os.path.join(script_directory, "configs", "lvcd.yaml")
        config = OmegaConf.load(config_path)
        config.model.params.drop_first_stage_model = False
        config.model.params.init_from_unet = False

        if use_xformers:
            config.model.params.network_config.params.spatial_transformer_attn_type = 'softmax-xformers'
            config.model.params.controlnet_config.params.spatial_transformer_attn_type = 'softmax-xformers'
            config.model.params.conditioner_config.params.emb_models[3].params.encoder_config.params.ddconfig.attn_type = 'vanilla-xformers'
        else:
            config.model.params.network_config.params.spatial_transformer_attn_type = 'softmax'
            config.model.params.controlnet_config.params.spatial_transformer_attn_type = 'softmax'
            config.model.params.conditioner_config.params.emb_models[3].params.encoder_config.params.ddconfig.attn_type = 'vanilla'

        config.model.params.ckpt_path = svd_model_path
        config.model.params.control_model_path = lvcd_path

        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval().requires_grad_(False)

        model.model.to(dtype)
        model.control_model.to(dtype)
        model.eval()
        model = model.requires_grad_(False)

        lvcd_pipe = {
            "model": model,
            "dtype": dtype,
        }

        return (lvcd_pipe,)
    
class LVCDSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "LVCD_pipe": ("LVCDPIPE",),
                "ref_images": ("IMAGE",),
                "sketch_images": ("IMAGE",),
                "num_frames": ("INT", {"default": 19, "min": 15, "max": 100, "step": 1}),
                "num_steps": ("INT", {"default": 25, "min": 1, "max": 100, "step": 1}),
                "fps_id": ("INT", {"default": 6, "min": 1, "max": 100, "step": 1}),
                "motion_bucket_id": ("INT", {"default": 160, "min": 0, "max": 1000, "step": 1}),
                "cond_aug": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "overlap": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1}),
                "prev_attn_steps": ("INT", {"default": 25, "min": 1, "max": 100, "step": 1}),
                "seed": ("INT", {"default": 123, "min": 0, "max": 2**32, "step": 1}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LVCDPIPE", "SVDSAMPLES",)
    RETURN_NAMES = ("LVCD_pipe", "samples",)
    FUNCTION = "loadmodel"
    CATEGORY = "ComfyUI-LVCDWrapper"

    def loadmodel(self, LVCD_pipe, ref_images, sketch_images, num_frames, num_steps, fps_id, motion_bucket_id, cond_aug, overlap, 
                  prev_attn_steps, seed, keep_model_loaded):
        
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.soft_empty_cache()

        model = LVCD_pipe["model"]

        inp = argparse.ArgumentParser()
        B, H, W, C = ref_images.shape
        inp.resolution = [H, W]

        inp.imgs = []
        inp.skts = []

        ref_images = ref_images.permute(0, 3, 1, 2).to(device) * 2 - 1
        for ref_img in ref_images:
            print(ref_img.shape)
            inp.imgs.append(ref_img.unsqueeze(0))
        sketch_images = sketch_images.permute(0, 3, 1, 2).to(device)
        for skt in sketch_images:
            inp.skts.append(skt.unsqueeze(0))

        arg = argparse.ArgumentParser()

        arg.ref_mode = 'prevref'
        arg.num_frames = num_frames
        arg.num_steps = num_steps
        arg.overlap = overlap
        arg.prev_attn_steps = prev_attn_steps
        arg.scale = [1.0, 1.0]
        arg.seed = seed
        arg.fps_id = fps_id
        arg.motion_bucket_id = motion_bucket_id
        arg.cond_aug = cond_aug

        model.to(device)
        model.control_model.to(device)
        samples = sample_video(model, device, inp, arg, verbose=True)
        if not keep_model_loaded:
            model.to(offload_device)
            model.control_model.to(offload_device)

        return (LVCD_pipe, samples)
    
class LVCDDecoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "LVCD_pipe": ("LVCDPIPE",),
                "samples": ("SVDSAMPLES",),
                "decoding_t": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "decoding_olap": ("INT", {"default": 3, "min": 0, "max": 100, "step": 1}),
                "decoding_first": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images", )
    FUNCTION = "loadmodel"
    CATEGORY = "ComfyUI-LVCDWrapper"

    def loadmodel(self, LVCD_pipe, samples, decoding_t, decoding_olap, decoding_first):
        
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.soft_empty_cache()

        model = LVCD_pipe["model"]

        arg = argparse.ArgumentParser()

        arg.decoding_t = decoding_t
        arg.decoding_olap = decoding_olap
        arg.decoding_first = decoding_first

        model.first_stage_model.to(device)
        frames = decode_video(model, device, samples, arg)
        model.first_stage_model.to(offload_device)

        min_value = frames.min()
        max_value = frames.max()

        frames = (frames - min_value) / (max_value - min_value)
        frames = frames.permute(0, 2, 3, 1).cpu().float()
        

        return (frames,)

NODE_CLASS_MAPPINGS = {
    "LoadLVCDModel": LoadLVCDModel,
    "LVCDSampler": LVCDSampler,
    "LVCDDecoder": LVCDDecoder,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadLVCDModel": "Load LVCD Model",
    "LVCDSampler": "LVCD Sampler",
    "LVCDDecoder": "LVCD Decoder",
    }
