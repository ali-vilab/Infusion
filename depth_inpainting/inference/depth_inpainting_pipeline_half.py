import os 
from typing import Any, Dict, Union
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
import cv2
from diffusers.utils import BaseOutput
from transformers import CLIPTextModel, CLIPTokenizer
import sys
from utils.image_util import resize_max_res,chw2hwc,colorize_depth_maps,create_point_cloud,write_ply_mask,write_ply,Disparity_Normalization_mask,resize_max_res_tensor
from utils.colormap import kitti_colormap

class DepthPipelineOutput(BaseOutput):
    """
    Output class for Marigold monocular depth prediction pipeline.

    Args:
        depth_np (`np.ndarray`):
            Predicted depth map, with depth values in the range of [0, 1].
        depth_colored (`PIL.Image.Image`):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """
    depth_np: np.ndarray
    depth_colored: Image.Image
    uncertainty: Union[None, np.ndarray]


class DepthEstimationInpaintPipeline(DiffusionPipeline):
    # two hyper-parameters
    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215
    
    def __init__(self,
                 unet:UNet2DConditionModel,
                 vae:AutoencoderKL,
                 scheduler:DDIMScheduler,
                 text_encoder:CLIPTextModel,
                 tokenizer:CLIPTokenizer,
                 ):
        super().__init__()
            
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.empty_text_embed = None
        
        # self.current_dtype = torch.float16
        
        
    
    
    @torch.no_grad()
    def __call__(self,
                 input_image_path:str,
                 denosing_steps: int =20,
                 processing_res: int = 768,
                 match_input_res:bool =True,
                 batch_size:int =0,
                 color_map: str="Spectral",
                 show_progress_bar:bool = True,
                 ensemble_kwargs: Dict = None,
                 depth_numpy = None,
                 mask = None,
                 path_to_save = None,
                 c2w= None,
                 intri=None,
                 colors_png=None,
                 use_mask=True,
                 blend=True

                 ) -> DepthPipelineOutput:
        
        # inherit from thea Diffusion Pipeline
        device = self.device
        input_image = Image.open(input_image_path)
        rgb_name_base = os.path.splitext(os.path.basename(input_image_path))[0]
        input_size = input_image.size
        size_w = input_size[0]
        size_h = input_size[1]
        # adjust the input resolution.
        if not match_input_res:
            assert (
                processing_res is not None                
            )," Value Error: `resize_output_back` is only valid with "
        
        assert processing_res >=0
        assert denosing_steps >=1
        
        # --------------- Image Processing ------------------------
        # Resize image
        if processing_res >0:
            input_image = resize_max_res(
                input_image, max_edge_resolution=processing_res
            ) 
        
        
        # Convert the image to RGB, to 1. reomve the alpha channel.
        input_image = input_image.convert("RGB")
        image = np.array(input_image)
        

        # Normalize RGB Values.
        rgb = np.transpose(image,(2,0,1))
        rgb_norm = rgb / 255.0
        rgb_norm = torch.from_numpy(rgb_norm).to(self.dtype)
        rgb_norm = rgb_norm.to(device)
        
        rgb_norm = rgb_norm.half()
        

        assert rgb_norm.min() >= 0.0 and rgb_norm.max() <= 1.0
        
        # ----------------- predicting depth -----------------
        duplicated_rgb = torch.stack([rgb_norm])
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        
        # find the batch size
        if batch_size>0:
            _bs = batch_size
        else:
            _bs = 1
        
        single_rgb_loader = DataLoader(single_rgb_dataset,batch_size=_bs,shuffle=False)
        
        # predicted the depth
        depth_pred_ls = []
        
        if show_progress_bar:
            iterable_bar = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable_bar = single_rgb_loader
        for batch in iterable_bar:
            (batched_image,)= batch  # here the image is still around 0-1
            depth_pred_raw,max_value,min_value,downscale_factor = self.single_infer(
                input_rgb=batched_image,
                depth_numpy = depth_numpy,
                mask = mask,
                num_inference_steps=denosing_steps,
                show_pbar=show_progress_bar,
                blend=blend
            )
            depth_pred_ls.append(depth_pred_raw.detach().clone())
        
        depth_preds = torch.concat(depth_pred_ls, axis=0).squeeze() #(10,224,768)
        torch.cuda.empty_cache()  # clear vram cache for ensembling
        

      
        depth_pred = depth_preds
        pred_uncert = None

        # ----------------- Post processing -----------------
        depth_save=(depth_pred*(max_value-min_value)+min_value)/downscale_factor
        depth_save=depth_save.detach().cpu().numpy()
        depth_save = cv2.resize(depth_save.astype(float),(size_w,size_h))
        dis_path =os.path.join(
            path_to_save, f"{rgb_name_base}_depth_dis.npy"
        )
        np.save(dis_path,depth_save)
        depth_path =os.path.join(
            path_to_save, f"{rgb_name_base}_depth.npy"
        )
        depth=np.reciprocal(depth_save)
        np.save(depth_path,depth)
        intrinsics=np.load(intri)
        c2w = np.load(c2w)
        colors=cv2.resize((cv2.imread(colors_png)/255),(size_w, size_h)).astype(float).reshape(-1,3)
        mask_save = cv2.resize(mask,(size_w, size_h)).astype(bool)
        points = create_point_cloud(depth, intrinsics, c2w)
        if use_mask:
            ply_path =os.path.join(
            path_to_save, f"{rgb_name_base}_mask.ply"
            )
            write_ply_mask(points,colors,ply_path,mask_save)
        else:
            ply_path =os.path.join(
            path_to_save, f"{rgb_name_base}.ply"
            )
            write_ply(points,colors,ply_path)
        min_d = torch.min(depth_pred)
        max_d = torch.max(depth_pred)
        depth_pred = (depth_pred - min_d) / (max_d - min_d)

         # Convert to numpy
        depth_pred= depth_pred.cpu().numpy().astype(np.float32)

        # Resize back to original resolution
        if match_input_res:
            pred_img = Image.fromarray(depth_pred)
            pred_img = pred_img.resize(input_size)
            depth_pred = np.asarray(pred_img)

        # Clip output range: current size is the original size
        depth_pred = depth_pred.clip(0, 1)
        
        # colorization using the KITTI Color Plan.
        depth_pred_vis = depth_pred * 70
        disp_vis = 400/(depth_pred_vis+1e-3)
        disp_vis = disp_vis.clip(0,500)
    
        # Colorize
        depth_colored = colorize_depth_maps(
            depth_pred, 0, 1, cmap=color_map
        ).squeeze()  # [3, H, W], value in (0, 1)
        depth_colored = (depth_colored * 255).astype(np.uint8)
        depth_colored_hwc = chw2hwc(depth_colored)
        depth_colored_img = Image.fromarray(depth_colored_hwc)

        
        return DepthPipelineOutput(
            depth_np = depth_pred,
            depth_colored = depth_colored_img,
            uncertainty=pred_uncert,
        )
        
    
    def __encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device) #[1,2]
        # print(text_input_ids.shape)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype) #[1,2,1024]
        self.empty_text_embed = self.empty_text_embed.half()
    def get_timesteps(self, num_inference_steps, strength, device, denoising_start=None):
        # get the original timestep using init_timestep
        if denoising_start is None:
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
        else:
            t_start = 0

        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        # Strength is irrelevant if we directly request a timestep to start at;
        # that is, strength is determined by the denoising_start instead.
        if denoising_start is not None:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_start * self.scheduler.config.num_train_timesteps)
                )
            )
            timesteps = list(filter(lambda ts: ts < discrete_timestep_cutoff, timesteps))
            return torch.tensor(timesteps), len(timesteps)

        return timesteps, num_inference_steps - t_start
        
    @torch.no_grad()
    def single_infer(self,input_rgb:torch.Tensor,
                     depth_numpy: np.ndarray,
                     mask: np.ndarray,
                     num_inference_steps:int,
                     show_pbar:bool,
                     blend:bool,):
        
        
        device = input_rgb.device
        
        # Set timesteps: inherit from the diffuison pipeline
        self.scheduler.set_timesteps(num_inference_steps, device=device) 
        timesteps = self.scheduler.timesteps  # [T]
        
        # encode image
        rgb_latent = self.encode_RGB(input_rgb) # 1/8 Resolution with a channel nums of 4. 

        # resize and normalize
        h=input_rgb.shape[2]
        w = input_rgb.shape[3]
        mask = torch.from_numpy(cv2.resize(mask,(w,h))).to(device)
       
        zero_indices = torch.nonzero(mask== 0)
        disparity = torch.from_numpy(depth_numpy).to(device)
        left_disparity_resized,downscale_factor = resize_max_res_tensor(disparity.unsqueeze(0).unsqueeze(0),is_disp=True) 
        min_value = torch.min(left_disparity_resized.squeeze()[zero_indices[:, 0], zero_indices[:, 1]])
        max_value = torch.max(left_disparity_resized.squeeze()[zero_indices[:, 0], zero_indices[:, 1]])
        left_disparity_resized_normalized = Disparity_Normalization_mask(left_disparity_resized, min_value,max_value)
        left_disparity_resized_normalized.squeeze()[mask==1]=0
        mask_disparity =left_disparity_resized_normalized.repeat(1,3,1,1).half()
        mask_disparity = self.encode_depth(mask_disparity)
        mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(mask.shape[0] // 8, mask.shape[1] // 8)).half()
        mask_blend=mask
        
        # Initial depth map (Guassian noise)
        timesteps_add,_=self.get_timesteps(num_inference_steps, 1.0, device, denoising_start=None)
        left_disparity_resized_normalized_no_mask = Disparity_Normalization_mask(left_disparity_resized, min_value,max_value)
        depth_latents_no_mask=left_disparity_resized_normalized_no_mask.repeat(1,3,1,1)

        disp_latents = self.encode_depth(depth_latents_no_mask.half())
        noise = torch.randn_like(disp_latents)
        depth_latent = self.scheduler.add_noise(disp_latents, noise, timesteps_add[:1])
        depth_latent = depth_latent.half()
        
        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.__encode_empty_text()
            
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        )  # [B, 2, 1024]
        
        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            unet_input = torch.cat([rgb_latent, depth_latent, mask, mask_disparity], dim=1)   
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            depth_latent = self.scheduler.step(noise_pred, t, depth_latent).prev_sample.to(self.dtype)
            if blend:
            # Blend diffusion https://arxiv.org/abs/2111.14818
                if i < len(timesteps) - 1:
                    noise_timestep = timesteps_add[i + 1]
                    disp_latent_step = self.scheduler.add_noise(
                        disp_latents, noise, torch.tensor([noise_timestep])
                    )
                    init_mask = F.interpolate(mask_blend,
                                    size=(h//8,w//8),mode='bilinear',
                                    align_corners=False).repeat(1,4,1,1).cuda().float()
                    init_mask = mask.repeat(1,4,1,1).float()


                    depth_latent = (1 - init_mask) * disp_latent_step  + init_mask * depth_latent
                    depth_latent = depth_latent.half()


        torch.cuda.empty_cache()
        depth = self.decode_depth(depth_latent)
        depth = torch.clip(depth, -1.0, 1.0)
        # shift to [0, 1]
        depth = (depth + 1.0) / 2.0
        return depth, max_value, min_value,downscale_factor
        
    
    def encode_RGB(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """

        
        # encode
        h = self.vae.encoder(rgb_in)

        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        
        return rgb_latent
    
    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # scale latent
        depth_latent = depth_latent / self.depth_latent_scale_factor
        
        depth_latent = depth_latent.half()
        # decode
        try:
            z = self.vae.post_quant_conv(depth_latent)
            stacked = self.vae.decoder(z)
        except:
            stacked = self.vae.decode(depth_latent)
        # mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean


    def encode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # scale latent
        h_disp = self.vae.encoder(depth_latent)
        moments_disp = self.vae.quant_conv(h_disp)
        mean_disp, logvar_disp = torch.chunk(moments_disp, 2, dim=1)
        disp_latents = mean_disp *self.depth_latent_scale_factor
        return disp_latents


