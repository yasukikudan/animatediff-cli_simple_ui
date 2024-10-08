# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union
import PIL
import cv2

import numpy as np
import torch
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL,ControlNetModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import (
    BaseOutput,
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
)


from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from packaging import version
from tqdm.rich import tqdm
from transformers import CLIPImageProcessor, CLIPTokenizer

from animatediff.models.clip import CLIPSkipTextModel
from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.context import get_context_scheduler, get_total_steps
from animatediff.utils.model import nop_train

logger = logging.getLogger(__name__)


@dataclass
class AnimationPipelineImg2ImgOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]

def ensure_one_dimension(t):
    if t.dim() == 0:
        return t.unsqueeze(0)
    elif t.dim() == 1:
        return t
    

def create_embedding_from_prompt(prompt_text, tokenizer, text_encoder, clip_skip, device, max_token_chunk_size=75, logger=None):
    # Process the prompt text and tokenize it
    prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    prompt_embeds_list = []
    total_token_count = len(prompt_tokens)
    current_token_index = 0

    while current_token_index < total_token_count:
        chunk_length = min(total_token_count - current_token_index, max_token_chunk_size)
        chunk=None
        if current_token_index>0:
            if chunk_length < max_token_chunk_size:
                chunk= prompt_tokens[(total_token_count -1)- max_token_chunk_size : total_token_count-1]
        if chunk is None:
            chunk = prompt_tokens[current_token_index : current_token_index + max_token_chunk_size]
        # Pad the chunk to ensure consistent length
        padding_length = max_token_chunk_size - len(chunk)
        chunk += [tokenizer.pad_token_id] * padding_length  # Padding with the tokenizer's pad token ID
        text_input_ids = torch.tensor([chunk]).to(device)

        if (
            hasattr(text_encoder.config, "use_attention_mask")
            and text_encoder.config.use_attention_mask
        ):
            attention_mask = torch.ones_like(text_input_ids).to(device)
        else:
            attention_mask = None

        chunk_embeds = text_encoder(
            text_input_ids,
            attention_mask=attention_mask,
            clip_skip=clip_skip,
        )
        prompt_embeds_list.append(chunk_embeds[0])

        current_token_index += max_token_chunk_size
    print(len(prompt_embeds_list))
    #(prompt_embeds_list[1].size())
    prompt_embeds = torch.cat(prompt_embeds_list).mean(dim=0).unsqueeze(0)
    print('test',prompt_embeds.size())

    return prompt_embeds



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


def auto_reshape_tensor(input_tensor):
    """
    Automatically reshape the tensor by merging the 0th and 2nd dimensions.
    
    Parameters:
    - input_tensor: The tensor to reshape.
    
    Returns:
    - reshaped_tensor: Tensor with merged 0th and 2nd dimensions.
    - original_shape: Original shape of the input tensor.
    """
    original_shape = input_tensor.shape
    merged_dim = original_shape[0] * original_shape[2]
    reshaped_tensor = input_tensor.view(merged_dim, original_shape[1], *original_shape[3:])
    return reshaped_tensor

def revert_auto_reshape(reshaped_tensor, split_factor=2):
    """
    Revert the reshaped tensor to its original shape by splitting the 0th dimension.
    
    Parameters:
    - reshaped_tensor: The tensor that was reshaped by merging the 0th and 2nd dimensions.
    - split_factor: The factor by which to split the 0th dimension. Default is 2.
    
    Returns:
    - Tensor reverted to its original shape.
    """
    original_shape = (split_factor, reshaped_tensor.shape[1], reshaped_tensor.shape[0] // split_factor, *reshaped_tensor.shape[2:])
    return reshaped_tensor.view(*original_shape)



#description: "This pipeline allows you to animate images using text prompts. It is based on the paper"
class AnimationPipelineImg2Img(DiffusionPipeline, TextualInversionLoaderMixin):
    _optional_components = ["feature_extractor"]

    vae: AutoencoderKL
    text_encoder: CLIPSkipTextModel
    tokenizer: CLIPTokenizer
    unet: UNet3DConditionModel
    controlnet: ControlNetModel
    feature_extractor: CLIPImageProcessor
    scheduler: Union[
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        LMSDiscreteScheduler,
        PNDMScheduler,
    ]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPSkipTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        controlnet: ControlNetModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        feature_extractor: CLIPImageProcessor,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding.

        When this option is enabled, the VAE will split the input tensor into tiles to compute decoding and encoding in
        several steps. This is useful to save a large amount of memory and to allow the processing of larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        hook = None
        for cpu_offloaded_model in [self.text_encoder, self.unet, self.vae]:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        if self.safety_checker is not None:
            _, hook = cpu_offload_with_hook(self.safety_checker, device, prev_module_hook=hook)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt: int = 1,
        do_classifier_free_guidance: bool = False,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: int = 1,
    ):
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        batch_size = len(prompt) if isinstance(prompt, list) else 1

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            prompt_embeds = create_embedding_from_prompt(prompt, self.tokenizer, self.text_encoder, clip_skip, device)
            """            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
                clip_skip=clip_skip,
            )"""
            #prompt_embeds = prompt_embeds[0]

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_input_ids = uncond_input.input_ids

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input_ids.to(device),
                attention_mask=attention_mask,
                clip_skip=clip_skip,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_videos_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def decode_latents(self, latents: torch.Tensor):
        video_length = latents.shape[2]
        latents = 1 / self.vae.config.scaling_factor * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in range(latents.shape[0]):
            video.append(
                self.vae.decode(latents[frame_idx : frame_idx + 1].to(self.vae.device, self.vae.dtype)).sample
            )
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        video_length,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=self.unet.device, dtype=dtype)
        else:
            latents = latents

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents.to(device, dtype)
    
    #description: "This pipeline allows you to animate images using text prompts. It is based on the paper"
    def prepare_latents_image(self, image, timestep, batch_size,video_length, num_images_per_prompt,dtype,device,generator=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            init_latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                init_latents = [
                    self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = self.vae.encode(image).latent_dist.sample(generator)

            init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        
        init_latents=init_latents.unsqueeze(2).repeat(1, 1,video_length ,1, 1)
        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # get latents
        if timestep is not None:
            init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
            latents = init_latents
        else:
            latents = init_latents + noise

        return latents
    
    def prepare_initial_latents(self, images, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None, strength=None):
        if not isinstance(images, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`images` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(images)}"
            )

        if isinstance(images, (torch.Tensor, PIL.Image.Image)):
            images = [images]

        initial_latents = []
        for image in images:
            image = image.to(device=device, dtype=dtype)
            batch_size = batch_size * num_images_per_prompt

            if image.shape[1] == 4:
                init_latents = image
            else:
                init_latents = self.vae.encode(image).latent_dist.sample(generator)
                init_latents = self.vae.config.scaling_factor * init_latents

            initial_latents.append(init_latents)

        print("複数枚画像を入力した場合の次元数",)
        init_latents = torch.cat(initial_latents, dim=2)
        init_latents = init_latents.unsqueeze(2).repeat(1, 1,len(images), 1, 1)
        print("複数枚画像を入力した場合の次元数", init_latents.size())
        return init_latents
    
    def prepare_controlnet_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    def get_latents_for_step(self, initial_latents, timestep, partial_latents=None,dtype=None,device=None,generator=None):
        shape = initial_latents.shape
        timestep=ensure_one_dimension(timestep)
        print('初期化ベクトルサイズ ',shape,timestep)
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self.scheduler.add_noise(initial_latents, noise, timestep)

        if partial_latents is not None:
            shape_partial = partial_latents.shape
            
            # Ensure that the partial_latents is not longer than the latents
            if shape_partial[2] >= shape[2]:
                # Replace the beginning of the partial_latents with the latents
                partial_latents[:, :, :shape[2], :, :] = latents
                latents = partial_latents  # Update latents to be the modified partial_latents

        return latents


    def adjust_latents_for_step(self, initial_latents, timestep, partial_latents, coefficient, dtype=None, device=None, generator=None):
        if not 0 <= coefficient <= 1:
            raise ValueError("coefficient should be in the range [0, 1]")

        shape = initial_latents.shape
        timestep=ensure_one_dimension(timestep)
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self.scheduler.add_noise(initial_latents, noise, timestep)

        if partial_latents is not None:

            # Compute the mean of partial_latents along the third dimension
            mean_partial_latents = partial_latents.mean(dim=2, keepdim=True)

            # Compute the difference between the mean of partial_latents and the initial_latents at index 0 along the third dimension
            diff = mean_partial_latents - initial_latents[:, :, 0:1, :, :]

            # Add the weighted difference to the partial_latents
            partial_latents=partial_latents+(coefficient * diff)

            # Replace the beginning of the latents with the adjusted partial_latents
            latents=partial_latents

        return latents


    
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        print('init',init_timestep)

        t_start = max(num_inference_steps - init_timestep, 0)
        print(t_start)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        print('steps',t_start * self.scheduler.order,self.scheduler.order)
        print(timesteps)

        return timesteps, num_inference_steps - t_start
    




    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        image: Optional[Union[torch.Tensor, PIL.Image.Image, List[Union[torch.Tensor, PIL.Image.Image]]]] = None,
        strength: Optional[float] = None,
        reference_image: Optional[Union[torch.Tensor, PIL.Image.Image, List[Union[torch.Tensor, PIL.Image.Image]]]] = None,
        image_guide: Optional[float] = None,
        video_length: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        context_frames: int = -1,
        context_stride: int = 3,
        context_overlap: int = 4,
        context_schedule: str = "uniform",
        clip_skip: int = 1,
        **kwargs,
    ):
        guess_mode=False
        print('画像の枚数',len(image))
        image=image[0:video_length]
    
        print('画像の枚数',len(image))
        print('画像の枚数',len(reference_image))
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 16 frames is max reliable number for one-shot mode, so we use sequential mode for longer videos
        sequential_mode = video_length is not None and video_length > 48

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # Define call parameters
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        latents_device = torch.device("cpu") if sequential_mode else device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_videos_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_skip,
        )
        print("各次元の要素数:", prompt_embeds.size(),batch_size)


        #strength=0.95
        #image_guide=0.75
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=latents_device)
        timesteps = self.scheduler.timesteps#*strength
        #print(timesteps,num_inference_steps)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_videos_per_prompt)
        
        #img2img guidenace step
        image_guide_step=int(len(timesteps))-int(len(timesteps)*image_guide)
        print(timesteps,num_inference_steps,latent_timestep)
        print("潜在変数の次元数:",latent_timestep.size())

      # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels

        #(batch_size,num_videos_per_prompt,video_length)
        print('データ生成',len(image))

        initial_image=self.image_processor.preprocess([image[0]])
        initial_latents2=self.prepare_initial_latents(
            initial_image,latent_timestep,batch_size,num_videos_per_prompt,self.vae.dtype, device, generator
        )
        del initial_image
        print("初期化ベクトルサイズ initial_latents2 ",initial_latents2.size())

        latents=[]
        if len(image)>1:
            for i,im in enumerate(image):
                im = self.image_processor.preprocess([im])
                latents.append(self.prepare_latents_image(
                    im,latent_timestep,batch_size,1,num_videos_per_prompt,self.vae.dtype, device, generator
                ))
            print("初期化ベクトルサイズ latents ",latents[0].size())
            latents=torch.cat(latents,dim=2)
            print("初期化ベクトルサイズ latents ",latents.size())
        
        else:
            im = self.image_processor.preprocess([image[0]])
            latents=self.prepare_latents_image(im,latent_timestep,batch_size,video_length,num_videos_per_prompt,self.vae.dtype, device, generator)
        
        #canny_image=[apply_canny(im) for im in image]
        #controlnet_image=self.prepare_controlnet_image(canny_image, width, height, batch_size, num_videos_per_prompt, device, self.controlnet.dtype, do_classifier_free_guidance, guess_mode)

        #del image
        #del canny_image
        
        reference_image = self.image_processor.preprocess(reference_image)
        initial_latents=self.prepare_latents_image(
               reference_image,None,batch_size,video_length,num_videos_per_prompt,self.vae.dtype, device, generator
        )
        del reference_image
  
                #デバック用に追加
        print("画像読み込みの次元数:", initial_latents.size())
        print(context_frames)


        print(batch_size,num_videos_per_prompt,num_channels_latents,video_length)

        #デバック用に追加
        print("各次元の要素数:", latents.size())

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.5 - Infinite context loop shenanigans
        context_scheduler = get_context_scheduler(context_schedule)
        total_steps = get_total_steps(
            context_scheduler,
            timesteps,
            num_inference_steps,
            latents.shape[2],
            context_frames,
            context_stride,
            context_overlap,
        )
        print("totals",total_steps)

        # 6.6 - ControlNet
        #control_guidance_start=[0.0]
        #control_guidance_end = [1.0]
        #controlnet_conditioning_scale=[0.7]
        # = []
        #for i in range(len(timesteps)):
        #    keeps = [
        #        1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
        #        for s, e in zip(control_guidance_start, control_guidance_end)
        #    ]
        #    controlnet_keep.append(keeps[0] if isinstance(self.controlnet, ControlNetModel) else keeps)


        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=total_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                print(initial_latents.size(),latents.size(),i,t)

                #img2img guidenace step latents update
                print(i)
                #if image_guide_step>=i:
                    #print(i,"重み更新",image_guide_step)
                    #latents =self.adjust_latents_for_step(initial_latents,t,latents,0.05*(image_guide_step/len(timesteps)),dtype=self.vae.dtype, device=device, generator=generator)
                    #=(self.get_latents_for_step(initial_latents2,t,latents,dtype=self.vae.dtype, device=device, generator=generator)*0.5)+(latents*0.5)



                           # controlnet(s) inference

                                
 

                if guess_mode and do_classifier_free_guidance:
                    # Infered ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                noise_pred = torch.zeros(
                    (latents.shape[0] * (2 if do_classifier_free_guidance else 1), *latents.shape[1:]),
                    device=latents.device,
                    dtype=latents.dtype#,
                    #down_block_additional_residuals=down_block_res_samples,
                    #=mid_block_res_sample
                )
                counter = torch.zeros(
                    (1, 1, latents.shape[2], 1, 1), device=latents.device, dtype=latents.dtype
                )

                for context in context_scheduler(
                    i, num_inference_steps, latents.shape[2], context_frames, context_stride, context_overlap
                ):
                    




                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        latents[:, :, context]
                        .to(device)
                        .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                    )
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)




                    # ref only part
                    ref_image_latents =(initial_latents[:, :, context]
                        .to(device)
                        .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                    )
                    noise = randn_tensor(
                        ref_image_latents.shape, generator=generator, device=device, dtype=ref_image_latents.dtype
                    )
                    ref_xt = self.scheduler.add_noise(
                        ref_image_latents,
                        noise,
                        t.reshape(
                            1,
                        ),
                    )
                    ref_xt = self.scheduler.scale_model_input(ref_xt, t)
                    print("ref_xt",ref_xt.size())
                    print("latent_model_input",latent_model_input.size())
                    MODE = "write"
                    self.unet(
                        ref_xt.to(self.unet.device, self.unet.dtype),
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )

                    """"                    if guess_mode and do_classifier_free_guidance:
                        # Infer ControlNet only for the conditional batch.
                        control_model_input = auto_reshape_tensor(latent_model_input)
                        control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                        controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                    else:
                        control_model_input = auto_reshape_tensor(latent_model_input)
                        controlnet_prompt_embeds = prompt_embeds

                    if isinstance(controlnet_keep[i], list):
                        cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                    else:
                        controlnet_cond_scale = controlnet_conditioning_scale
                        if isinstance(controlnet_cond_scale, list):
                            controlnet_cond_scale = controlnet_cond_scale[0]
                        cond_scale = controlnet_cond_scale * controlnet_keep[i]
                    
                    #print("controlnet",self.controlnet)
                    print("controlnet",self.controlnet.device, self.controlnet.dtype)
                    print("test",latents.device, latents.dtype)
                    print(latent_model_input.size())
                    print(control_model_input.size())
                    print(controlnet_prompt_embeds.size())
                    print(context)
                    print(controlnet_image[:, :, context].size())
                    self.controlnet.to(latents.device)
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        control_model_input.to(self.controlnet.device, self.controlnet.dtype),
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds.to(self.controlnet.device, self.controlnet.dtype),
                        controlnet_cond=controlnet_image[ [i*2 for i in context] + [i*2 + 1 for i in context]],
                        conditioning_scale=cond_scale,
                        guess_mode=guess_mode,
                        return_dict=False,
                    )
                    self.controlnet.to(torch.device("cpu"))
                    print(down_block_res_samples.size())
                    print(mid_block_res_sample.size())
                    """
                    
                    MODE = "read"
                    # predict the noise residual
                    pred = self.unet(
                        latent_model_input.to(self.unet.device, self.unet.dtype),
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        #down_block_res_samples=down_block_res_samples,
                        #mid_block_res_sample=mid_block_res_sample,
                        return_dict=False,
                    )[0]

                    pred = pred.to(dtype=latents.dtype, device=latents.device)
                    noise_pred[:, :, context] = noise_pred[:, :, context] + pred
                    counter[:, :, context] = counter[:, :, context] + 1
                    progress_bar.update()

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = (noise_pred / counter).chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    model_output=noise_pred.to(latents_device),
                    timestep=t,
                    sample=latents.to(latents_device),
                    **extra_step_kwargs,
                    return_dict=False,
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # Return latents if requested (this will never be a dict)
        if not output_type == "latent":
            video = self.decode_latents(latents)
        else:
            video = latents

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return video

        return AnimationPipelineImg2ImgOutput(videos=video)

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def freeze(self):
        logger.debug("Freezing pipeline...")
        _ = self.unet.eval()
        self.unet = self.unet.requires_grad_(False)
        self.unet.train = nop_train

        _ = self.text_encoder.eval()
        self.text_encoder = self.text_encoder.requires_grad_(False)
        self.text_encoder.train = nop_train

        _ = self.vae.eval()
        self.vae = self.vae.requires_grad_(False)
        self.vae.train = nop_train
