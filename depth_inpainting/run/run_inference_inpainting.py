import argparse
import os
import logging

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
import cv2
import sys
sys.path.append("../")
from inference.depth_inpainting_pipeline_half import DepthEstimationInpaintPipeline
from utils.seed_all import seed_all
import matplotlib.pyplot as plt


from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from diffusers import DiffusionPipeline, ConsistencyDecoderVAE

from transformers import CLIPTextModel, CLIPTokenizer




if __name__=="__main__":
    
    use_seperate = True
    
    logging.basicConfig(level=logging.INFO)
    
    '''Set the Args'''
    parser = argparse.ArgumentParser(
        description="Run MonoDepth Estimation using Stable Diffusion."
    )
    parser.add_argument(
        "--c2w",
        type=str,
        default='None',
        help="Camera2world of the selected view.",
    ) 
    parser.add_argument(
        "--intri",
        type=str,
        default='None',
        help="Intrinsics of the selected view.",
    )
    parser.add_argument(
        "--input_rgb_path",
        type=str,
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--input_mask",
        type=str,
        required=True,
        help="Path to the mask that needs to be inpainted.",
    )
    parser.add_argument(
        "--input_depth_path",
        type=str,
        required=True,
        help="Path to the depth of original gaussians.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to depth inpainting model."
    )
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=20,
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed.",
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=768,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 768.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )

    # depth map colormap
    parser.add_argument(
        "--color_map",
        type=str,
        default="Spectral",
        help="Colormap used to render depth predictions.",
    )
    # other settings
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Inference batch size. Default: 0 (will be set automatically).",
    )
    parser.add_argument(
        "--use_mask",
        action="store_true",
        help="If true, only the inpainted part of the point cloud is stored.",
    )
    parser.add_argument(
        "--blend",
        action="store_true",
        help="If true, using blend diffusion inference method.",
    )
    args = parser.parse_args()
    input_image_path = args.input_rgb_path
    input_depth_path = args.input_depth_path
    input_mask = args.input_mask
    output_dir = args.output_dir
    denoise_steps = args.denoise_steps
    
    
    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res

    color_map = args.color_map
    seed = args.seed
    batch_size = args.batch_size
    
    if batch_size==0:
        batch_size = 1  # set default batchsize
    
    # -------------------- Preparation --------------------
    # Random seed
    if seed is None:
        import time

        seed = int(time.time())
    seed_all(seed)

    # Output directories
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")
    
    
    # -------------------Data----------------------------
    logging.info("Inference Image Path from {}".format(input_image_path))
    dtype = torch.float16
    
    
    if not use_seperate:
        pipe = DepthEstimationPipeline.from_pretrained(checkpoint_path, torch_dtype=dtype)
        print("Using Completed")
    else:
        
        vae = AutoencoderKL.from_pretrained(args.model_path,subfolder='vae',torch_dtype=dtype)
        scheduler = DDIMScheduler.from_pretrained(args.model_path,subfolder='scheduler',torch_dtype=dtype)
        text_encoder = CLIPTextModel.from_pretrained(args.model_path,subfolder='text_encoder',torch_dtype=dtype)
        tokenizer = CLIPTokenizer.from_pretrained(args.model_path,subfolder='tokenizer',torch_dtype=dtype)
        
        unet = UNet2DConditionModel.from_pretrained(args.model_path,subfolder="unet",
                                                    in_channels=13, sample_size=96,
                                                    low_cpu_mem_usage=False,
                                                    ignore_mismatched_sizes=True,
                                                    torch_dtype=dtype)
        
        pipe = DepthEstimationInpaintPipeline(unet=unet,
                                       vae=vae,
                                       scheduler=scheduler,
                                       text_encoder=text_encoder,
                                       tokenizer=tokenizer,
                                       )
        print("Using Seperated Modules")
    
    logging.info("loading pipeline whole successfully.")
    
    try:

        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass  # run without xformers

    pipe = pipe.to(device)

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        depth_numpy  = np.load(input_depth_path)
        mask = (cv2.imread(input_mask,cv2.IMREAD_GRAYSCALE)/255).astype(float)
        mask[mask>0.5]=1
        mask[mask<0.5]=0

        pipe_out = pipe(input_image_path,
             denosing_steps=denoise_steps,
             processing_res = processing_res,
             match_input_res = match_input_res,
             batch_size = batch_size,
             color_map = color_map,
             show_progress_bar = True,
             depth_numpy = depth_numpy,
             mask = mask,
             path_to_save = output_dir,
             c2w=args.c2w,
             intri=args.intri,
             colors_png=args.input_rgb_path,
             use_mask=args.use_mask,
             blend=args.blend
             )
        depth_colored: Image.Image = pipe_out.depth_colored
        # savd as npy
        rgb_name_base = os.path.splitext(os.path.basename(input_image_path))[0]
        pred_name_base = rgb_name_base + "_pred"

        # Colorize
        colored_save_path = os.path.join(
            output_dir, f"{pred_name_base}_colored.png"
        )
        if os.path.exists(colored_save_path):
            logging.warning(
                f"Existing file: '{colored_save_path}' will be overwritten"
            )
        depth_colored.save(colored_save_path)