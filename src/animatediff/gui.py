import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional
import flet as ft

import json
import os

import torch
import typer
from diffusers.utils.logging import \
    set_verbosity_error as set_diffusers_verbosity_error
from rich.logging import RichHandler

from animatediff import __version__, console, get_dir
from animatediff.generate_gui import create_pipeline, run_inference
from animatediff.pipelines import AnimationPipeline, load_text_embeddings
from animatediff.settings import (CKPT_EXTENSIONS, InferenceConfig,
                                  ModelConfig, get_infer_config,
                                  get_model_config)
from animatediff.utils.model import checkpoint_to_pipeline, get_base_model
from animatediff.utils.pipeline import get_context_params, send_to_device
from animatediff.utils.util import path_from_cwd, save_frames, save_video

cli: typer.Typer = typer.Typer(
    context_settings=dict(help_option_names=["-h", "--help"]),
    rich_markup_mode="rich",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
data_dir = get_dir("data")
checkpoint_dir = data_dir.joinpath("models/sd")
pipeline_dir = data_dir.joinpath("models/huggingface")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        RichHandler(console=console, rich_tracebacks=True),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

pipeline: Optional[AnimationPipeline] = None
last_model_path: Optional[Path] = None

set_diffusers_verbosity_error()


infer=get_infer_config()
print(infer)    
device: torch.device = torch.device('cuda')

pipeline=create_pipeline(
    base_model="data/models/huggingface/stable-diffusion-v1-5/",
    #base_model="data/models/sd/ar100.safetensors",
    controlnet_model="data/models/controlnet/control_v11p_sd15_canny",
    checkpoint_path="data/models/sd/ar100.safetensors",
    #checkpoint_path="data/models/sd/shemale.safetensors",
    motion_module_path="data/models/motion-module/mm_sd_v14.safetensors",
    scheduler_type="k_dpmpp_2m",
    use_xformers=False,
    infer_config=infer
)


pipeline = send_to_device(
        pipeline, device, freeze=True, force_half=False, compile=False
    )   

def main(page: ft.Page):
    def slider_changed(e):
        # This function will be called whenever a slider value changes
        pass

    def create_validator(target_type, default_value=None):
        def validator(e):
            try:
                # 指定された型にキャスト
                cast_value = target_type(e.control.value)
            except (ValueError, TypeError):
                # キャストできない場合、エラーメッセージを表示
                print(f"Invalid input: Not a valid {target_type.__name__}")
                if default_value is not None:
                    # デフォルト値が指定されていれば、コントロールの値をリセット
                    e.control.value = default_value
            else:
                # キャストできる場合、値を更新
                e.control.value = cast_value
        return validator



    def button_clicked(e):
        # Collecting the values from the input widgets and run inference
        global pipeline


        params = {
            'prompt': prompt_field.value,
            'n_prompt': n_prompt_field.value,
            'seed': seed_slider.value,
            'steps': steps_slider.value,
            'guidance_scale': guidance_scale_slider.value,
            'image': image_field.value,
            'strength': strength_slider.value,
            'canny_image': canny_image_field.value if len(canny_image_field.value)>0 else None,
            'reference_image':reference_image_field.value if len(reference_image_field.value)>0 else None,
            'image_guide': image_guide_slider.value,
            'width': width_slider.value,
            'height': height_slider.value,
            'duration': duration_slider.value,
            'idx': 0,
            'out_dir': out_dir_field.value,
            'context_frames': 16,
            'context_stride': 3,
            'context_overlap': 4,
            'context_schedule': None,
            'controlnet_conditioning_scale': controlnet_conditioning_scale_slider.value,
            'controlnet_conditioning_start': controlnet_conditioning_start_slider.value,
            'controlnet_conditioning_end': controlnet_conditioning_end_slider.value,
            'controlnet_conditioning_bias': controlnet_conditioning_bias_slider.value,
            'controlnet_preprocessing': None,
            'clip_skip': 2
        }
    
        # 現在のタイムスタンプを取得
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        
        # 出力ディレクトリのパスを取得
        out_dir = out_dir_field.value
        
        # JSONファイル名を作成
        json_filename = f'params_{timestamp}.json'
        
        # JSONファイルのフルパスを作成
        json_filepath = os.path.join(out_dir, json_filename)
        
        # パラメータをJSON形式でシリアライズし、ファイルに保存
        with open(json_filepath, 'w') as json_file:
            json.dump(params, json_file, indent=4)
        
        res=run_inference(
            pipeline=pipeline,
            prompt=prompt_field.value,
            n_prompt=n_prompt_field.value,
            seed=seed_slider.value,
            steps=steps_slider.value,
            guidance_scale=guidance_scale_slider.value,
            image=image_field.value,
            strength=strength_slider.value,
            canny_image=canny_image_field.value if len(canny_image_field.value)>0 else None,
            reference_image=reference_image_field.value if len(reference_image_field.value)>0 else None,
            image_guide=image_guide_slider.value,
            width=width_slider.value,
            height=height_slider.value,
            duration=duration_slider.value,
            idx=0,
            out_dir=out_dir_field.value,
            context_frames=16,
            context_stride=3,
            context_overlap=4,
            context_schedule=None,
            controlnet_conditioning_scale=controlnet_conditioning_scale_slider.value,
            controlnet_conditioning_start=controlnet_conditioning_start_slider.value,
            controlnet_conditioning_end=controlnet_conditioning_end_slider.value,
            controlnet_conditioning_bias=controlnet_conditioning_bias_slider.value,
            controlnet_preprocessing='canny',
            clip_skip=2
        )
        del res
        pipeline = send_to_device(
        pipeline, device, freeze=True, force_half=False, compile=False
        )   


    # Creating input widgets for each parameter with labels displaying the variable names and values
    prompt_field = ft.TextField(label="prompt")
    n_prompt_field = ft.TextField(label="n_prompt")
    seed_slider = ft.TextField(label='seed', on_change=create_validator(int),keyboard_type=ft.KeyboardType.NUMBER)
    steps_slider = ft.Slider(min=0, max=100, divisions=100,value=20, label="steps: {value}", on_change=create_validator(int))
    guidance_scale_slider = ft.Slider(min=0.0, max=20.0, divisions=40,value=7.5, label="guidance_scale: {value}", on_change=create_validator(float))
    image_field = ft.TextField(label="image")
    strength_slider = ft.Slider(min=0.0, max=1.0, divisions=100,value=1.0 ,label="strength:{value}%", on_change=create_validator(float))
    canny_image_field = ft.TextField(label="canny_image",value=None)
    reference_image_field = ft.TextField(label="reference_image",value=None)
    image_guide_slider = ft.Slider(min=0.0, max=1.0, divisions=100,value=0.0,label="image_guide: {value}%", on_change=create_validator(float))
    width_slider = ft.Slider(min=0, max=2048, divisions=256, label="width: {value}", value=512,    on_change=create_validator(int))
    height_slider = ft.Slider(min=0, max=2048, divisions=256, label="height: {value}",value=512,   on_change=create_validator(int))
    duration_slider = ft.Slider(min=16, max=512,divisions=62,label="duration: {value}",value=16, on_change=create_validator(int))
    out_dir_field = ft.TextField(label="out_dir",value="data_test")
    controlnet_conditioning_scale_slider = ft.Slider(min=0.0, max=1.0, divisions=100, value=0.0,label="controlnet_conditioning_scale: {value}", on_change=create_validator(float))
    controlnet_conditioning_start_slider = ft.Slider(min=0.0, max=1.0, divisions=100, value=0.0,label="controlnet_conditioning_start: {value}", on_change=create_validator(float))
    controlnet_conditioning_end_slider   = ft.Slider(min=0.0, max=1.0, divisions=100, value=0.1,label="controlnet_conditioning_end: {value}", on_change=create_validator(float))
    controlnet_conditioning_bias_slider  = ft.Slider(min=0.0, max=1.0, divisions=100, value=0.0,label="controlnet_conditioning_bias: {value}", on_change=create_validator(float))

    # Creating the submit button
    submit_button = ft.ElevatedButton(text="Submit", on_click=button_clicked)

    # Adding all the input widgets and the submit button to the page
    page.add(
        prompt_field, n_prompt_field, seed_slider, steps_slider, guidance_scale_slider,
        image_field, strength_slider, canny_image_field, reference_image_field, image_guide_slider,
        width_slider, height_slider, duration_slider, out_dir_field,
        controlnet_conditioning_scale_slider, controlnet_conditioning_start_slider,
        controlnet_conditioning_end_slider, controlnet_conditioning_bias_slider,submit_button
    )

# Running the application
ft.app(target=main,port=8550, view=ft.WEB_BROWSER)



"""
run_inference(
    pipeline=pipeline,
    prompt="anime,best quality,pov,1girl,solo,gay,milf,shemale,large penis,naked,abs,step mam,penis,erection,4k,depth_of_field,masterpiece,in sauna",
    n_prompt="worst quality,low quality,painting,sketch,flat color,monochrome,grayscale,ugly face,bad face,bad anatomy,deformed eyes,missing fingers,acnes,skin blemishes",
    seed=10000000000,
    steps=25,
    guidance_scale=7.5,
    image=None,
    strength=1.0,
    canny_image=None,
    reference_image=None,
    image_guide=0.0,
    width  = 512,
    height = 912,
    duration =32,
    idx      = 0,
    out_dir="data_test",
    context_frames   =16,
    context_stride   = 3,
    context_overlap  = 4,
    context_schedule = None,
    controlnet_conditioning_scale = 0.01,
    controlnet_conditioning_start = 0.1,
    controlnet_conditioning_end  = 0.2,
    controlnet_conditioning_bias  = 0.0,
    controlnet_preprocessing ="none",
    clip_skip  = 2
)
"""