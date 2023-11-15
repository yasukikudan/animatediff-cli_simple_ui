import logging
import re
from os import PathLike
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import PIL
import torch
from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionPipeline
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from animatediff import get_dir
from animatediff.models.clip import CLIPSkipTextModel
from animatediff.models.controlnet import ControlNetModel3D
from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines import (AnimationPipeline, AnimationPipelineImg2Img,
                                   AnimationPipelineImg2ImgControlnet,
                                   load_text_embeddings)
from animatediff.schedulers import get_scheduler
from animatediff.settings import InferenceConfig, ModelConfig
from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora
from animatediff.utils.model import (ensure_motion_modules,
                                     get_checkpoint_weights)
from animatediff.utils.util import save_video

logger = logging.getLogger(__name__)

data_dir = get_dir("data")
default_base_path = data_dir.joinpath("models/huggingface/stable-diffusion-v1-5")

re_clean_prompt = re.compile(r"[^\w\-, ]")



def load_images(image, width=None, height=None):
    """
    Load and optionally resize images from a given path or a list of paths.

    Parameters:
    - image: Single path (str or Path object) or list of paths.
    - width: Desired width of the image.
    - height: Desired height of the image.

    Returns:
    - List of PIL Image objects.
    """
    images = []

    def open_and_resize(img_path):
        img = PIL.Image.open(img_path).convert('RGB')
        if width is not None:
            if height is not None:
                img = img.resize((width, height))
        return img

    # Single path case
    if isinstance(image, (str, Path)):
        path = Path(image)

        if path.is_file():
            images.append(open_and_resize(path))
        elif path.is_dir():
            # Load image files in ascending order from the directory
            for img_path in sorted(path.glob('*')):
                if img_path.suffix in ['.jpg','.jpeg', '.png']:
                    images.append(open_and_resize(img_path))

    # List of paths case
    elif isinstance(image, list):
        for img_path in image:
            path_obj = Path(img_path)
            if path_obj.is_file() and path_obj.suffix in ['.jpg','.jpeg', '.png']:
                images.append(open_and_resize(path_obj))

    return images

def apply_canny(input_image, low_threshold=100, high_threshold=200):
    """
    Applies the Canny edge detection on the given image and then converts it to a 3-channel image.

    Parameters:
    - input_image: The input image to be processed.
    - low_threshold: The low threshold for the Canny edge detection. Default is 100.
    - high_threshold: The high threshold for the Canny edge detection. Default is 200.

    Returns:
    - control_image: The processed 3-channel image.
    """

    image_array = np.array(input_image)

    # グレースケールに変換
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    image = cv2.Canny(gray_image, low_threshold, high_threshold)
    return image


def convert_to_gray(input_image):

    input_image = input_image.convert('RGB')
    image_array = np.array(input_image)
    cv2_image=cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    return cv2_image


def convert_to_rgb(input_image):

    input_image = input_image.convert('RGB')
    image_array = np.array(input_image)
    #cv2_image=cv2.cvtColor(image_array, cv2.COLOR_RGB2R)
    return image_array


def create_pipeline(
    base_model: Union[str, PathLike] = default_base_path,
    model_config: ModelConfig = ...,
    infer_config: InferenceConfig = ...,
    use_xformers: bool = True,
) -> AnimationPipelineImg2ImgControlnet:
    """Create an AnimationPipeline from a pretrained model.
    Uses the base_model argument to load or download the pretrained reference pipeline model."""

    # make sure motion_module is a Path and exists
    logger.info("Checking motion module...")
    controlnet_model=model_config.controlnet
    motion_module = data_dir.joinpath(model_config.motion_module)
    if not (motion_module.exists() and motion_module.is_file()):
        # check for safetensors version
        motion_module = motion_module.with_suffix(".safetensors")
        if not (motion_module.exists() and motion_module.is_file()):
            # download from HuggingFace Hub if not found
            ensure_motion_modules()
        if not (motion_module.exists() and motion_module.is_file()):
            # this should never happen, but just in case...
            raise FileNotFoundError(f"Motion module {motion_module} does not exist or is not a file!")

    logger.info("Loading tokenizer...")
    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    logger.info("Loading text encoder...")
    text_encoder: CLIPSkipTextModel = CLIPSkipTextModel.from_pretrained(base_model, subfolder="text_encoder")
    logger.info("Loading VAE...")
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    logger.info("Loading UNet...")
    unet: UNet3DConditionModel = UNet3DConditionModel.from_pretrained_2d(
        pretrained_model_path=base_model,
        motion_module_path=motion_module,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    )
    logger.info("Loading ControlNet...")
    print(controlnet_model)
    controlnet=ControlNetModel3D.from_pretrained_2d(controlnet_model).to(dtype=torch.float16)
  #  controlnet=None
    feature_extractor = CLIPImageProcessor.from_pretrained(base_model, subfolder="feature_extractor")

    # set up scheduler
    sched_kwargs = infer_config.noise_scheduler_kwargs
    scheduler = get_scheduler(model_config.scheduler, sched_kwargs)
    logger.info(f'Using scheduler "{model_config.scheduler}" ({scheduler.__class__.__name__})')

    # Load the checkpoint weights into the pipeline
    if model_config.path is not None:
        model_path = data_dir.joinpath(model_config.path)
        logger.info(f"Loading weights from {model_path}")
        if model_path.is_file():
            logger.debug("Loading from single checkpoint file")
            unet_state_dict, tenc_state_dict, vae_state_dict = get_checkpoint_weights(model_path)
        elif model_path.is_dir():
            logger.debug("Loading from Diffusers model directory")
            temp_pipeline = StableDiffusionPipeline.from_pretrained(model_path)
            unet_state_dict, tenc_state_dict, vae_state_dict = (
                temp_pipeline.unet.state_dict(),
                temp_pipeline.text_encoder.state_dict(),
                temp_pipeline.vae.state_dict(),
            )
            del temp_pipeline
        else:
            raise FileNotFoundError(f"model_path {model_path} is not a file or directory")

        # Load into the unet, TE, and VAE
        logger.info("Merging weights into UNet...")
        _, unet_unex = unet.load_state_dict(unet_state_dict, strict=False)
        if len(unet_unex) > 0:
            raise ValueError(f"UNet has unexpected keys: {unet_unex}")
        tenc_missing, _ = text_encoder.load_state_dict(tenc_state_dict, strict=False)
        if len(tenc_missing) > 0:
            raise ValueError(f"TextEncoder has missing keys: {tenc_missing}")
        vae_missing, _ = vae.load_state_dict(vae_state_dict, strict=False)
        if len(vae_missing) > 0:
            raise ValueError(f"VAE has missing keys: {vae_missing}")
    else:
        logger.info("Using base model weights (no checkpoint/LoRA)")

    # enable xformers if available
    if use_xformers:
        logger.info("Enabling xformers memory-efficient attention")
        unet.enable_xformers_memory_efficient_attention()

    # I'll deal with LoRA later...

    logger.info("Creating AnimationPipeline...")
    pipeline = AnimationPipelineImg2ImgControlnet(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        scheduler=scheduler,
        feature_extractor=feature_extractor,
    )

    # Load TI embeddings
    load_text_embeddings(pipeline)

    return pipeline


def run_inference(
    pipeline: AnimationPipelineImg2ImgControlnet,
    prompt: str = ...,
    n_prompt: str = ...,
    seed: int = -1,
    steps: int = 25,
    guidance_scale: float = 7.5,
    image=None,
    strength=None,
    canny_image=None,
    reference_image=None,
    image_guide=None,
    width: int = 512,
    height: int = 512,
    duration: int = 16,
    idx: int = 0,
    out_dir: PathLike = ...,
    context_frames: int = -1,
    context_stride: int = 3,
    context_overlap: int = 4,
    context_schedule: str = None,
    controlnet_conditioning_scale = 0.01,
    controlnet_conditioning_start = 0.1,
    controlnet_conditioning_end  = 0.2,
    controlnet_conditioning_bias  = 0.0,
    controlnet_preprocessing = "none",
    clip_skip: int = 1,
    return_dict: bool = False,
):
    out_dir = Path(out_dir)  # ensure out_dir is a Path

    if seed != -1:
        torch.manual_seed(seed)
    else:
        seed = torch.seed()

    image=load_images(image,width= width,height= height)
    print('image',len(image))
    reference_image=load_images(reference_image,width= width,height= height)

    if canny_image is not None:
        canny_image=load_images(canny_image,width= width,height= height)
        if(controlnet_preprocessing=="canny"):
            canny_image=[apply_canny(image,low_threshold=50,high_threshold=100) for image in canny_image]
        elif(controlnet_preprocessing=="gray"):
            canny_image=[convert_to_gray(image) for image in canny_image]
        else:
            canny_image=[convert_to_rgb(image) for image in canny_image]
        print('canny_image',len(canny_image))


    pipeline_output = pipeline(
        prompt=prompt,
        negative_prompt=n_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        image=image,
        strength=strength,
        canny_image=canny_image,
        reference_image=reference_image,
        image_guide=image_guide,
        width=width,
        height=height,
        video_length=duration,
        return_dict=return_dict,
        context_frames=context_frames,
        context_stride=context_stride + 1,
        context_overlap=context_overlap,
        context_schedule=context_schedule,
        controlnet_conditioning_scale = controlnet_conditioning_scale,
        controlnet_conditioning_start = controlnet_conditioning_start,
        controlnet_conditioning_end   = controlnet_conditioning_end,
        controlnet_conditioning_bias  = controlnet_conditioning_bias,
        clip_skip=clip_skip,
    )
    logger.info("Generation complete, saving...")

    # Trim and clean up the prompt for filename use
    prompt_tags = [re_clean_prompt.sub("", tag).strip().replace(" ", "-") for tag in prompt.split(",")]
    prompt_str = "_".join((prompt_tags[:6]))

    # generate the output filename and save the video
    out_file = out_dir.joinpath(f"{idx:02d}_{seed}_{prompt_str}.gif")
    if return_dict is True:
        save_video(pipeline_output["videos"], out_file)
    else:
        save_video(pipeline_output, out_file)

    logger.info(f"Saved sample to {out_file}")
    return pipeline_output
