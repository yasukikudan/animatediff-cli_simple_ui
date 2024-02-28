import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional
import json
import flet as ft

import json
import os

import torch
import typer
from diffusers.utils.logging import \
    set_verbosity_error as set_diffusers_verbosity_error
from rich.logging import RichHandler

from animatediff import __version__, console, get_dir
from animatediff.generate_gui import create_pipeline, run_inference,refresh_pipeline
from animatediff.pipelines import AnimationPipeline, load_text_embeddings
from animatediff.settings import (CKPT_EXTENSIONS, InferenceConfig,
                                  ModelConfig, get_infer_config,
                                  get_model_config,get_pipeline_config)
from animatediff.utils.model import checkpoint_to_pipeline, get_base_model
from animatediff.utils.pipeline import get_context_params, send_to_device
from animatediff.utils.util import save_frames, save_video,encode_frames,save_animation,save_images
import gc
import random
import sys


cli: typer.Typer = typer.Typer(
    context_settings=dict(help_option_names=["-h", "--help"]),
    rich_markup_mode="rich",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
data_dir = get_dir("data")
checkpoint_dir = data_dir.joinpath("models/sd")
pipeline_dir = data_dir.joinpath("models/huggingface")

positive_prompt_default="(anime,masterpiece,best quality:1.1),"
negative_prompt_default="worst quality,low quality,bad anatomy,missing fingers,bad fingers,missing fingers,blurry,bokeh, blur,jpeg, gausian noise, block noise,cropped, lowres, text, multiple view,block noise,pixelated,unfocused,grainy"




pipeline: Optional[AnimationPipeline] = None
last_model_path: Optional[Path] = None

set_diffusers_verbosity_error()


infer=get_infer_config()
print(infer)    
device: torch.device = torch.device('cuda')



config = get_pipeline_config()

# パイプラインの作成に設定を使用
pipeline = create_pipeline(
    base_model=config["base_model"],
    controlnet_model=config["controlnet_model"],
    checkpoint_path=config["checkpoint_path"],
    motion_module_path=config["motion_module_path"],
    scheduler_type=config["scheduler_type"],
    use_xformers=config["use_xformers"],
    infer_config=infer  # 既存の変数をそのまま使用
)


pipeline = send_to_device(
        pipeline, device, freeze=True, force_half=False, compile=False
    )   



import io 
import re
import base64
def image_to_animetion_base64(images,fps):
    # メモリストリームを作成
    bytes_io = io.BytesIO()
    # WebPとしてメモリストリームに保存
    images[0].save(bytes_io, format='WEBP', save_all=True, append_images=images[1:], duration=int(1000/fps), loop=0)
    # メモリストリームをBase64にエンコード
    base64_encoded = base64.b64encode(bytes_io.getvalue()).decode('utf-8')
    bytes_io.close()
    return base64_encoded

#保存先の
def generate_output_filename(out_dir, idx, seed, prompt):
    re_clean_prompt = re.compile(r"[^a-zA-Z0-9]+")
    prompt_tags = [re_clean_prompt.sub("", tag).strip().replace(" ", "-") for tag in prompt.split(",")]
    prompt_str = "_".join((prompt_tags[:6]))
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(out_dir).joinpath((f"{idx:02d}_{seed}_{prompt_str}_{now}")[:100]+".webp")

#@profile
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

        

        #print("seed_field.value",seed_field.value,type(seed_field.value))
        if str(seed_field.value).isnumeric()==False:
            seed_field.value=str(random.randrange(sys.maxsize))

        params = {
            'prompt': prompt_field.value,
            'n_prompt': n_prompt_field.value,
            'seed': int(seed_field.value),
            'steps': int(steps_slider.value),
            'guidance_scale': guidance_scale_slider.value,
            'image': image_field.value,
            'strength': float(strength_slider.value)/100.0,
            'canny_image': canny_image_field.value if len(canny_image_field.value)>0 else None,
            'reference_image':reference_image_field.value if len(reference_image_field.value)>0 else None,
            'image_guide': float(image_guide_slider.value)/100.0,
            'width': int(width_slider.value),
            'height': int(height_slider.value),
            'duration': duration_slider.value,
            'idx': 0,
            'out_dir': out_dir_field.value,
            'context_frames': 16,
            'context_stride': 3,
            'context_overlap': 4,
            'context_schedule': schedule_dropdown.value,
            'controlnet_conditioning_scale':  float(controlnet_conditioning_scale_slider.value)/100.0,
            'controlnet_conditioning_start':  float(controlnet_conditioning_start_slider.value)/100.0,
            'controlnet_conditioning_end':    float(controlnet_conditioning_end_slider.value)/100.0,
            'controlnet_conditioning_bias':   float(controlnet_conditioning_bias_slider.value)/100.0,
            'controlnet_preprocessing': controlnet_preprocessing_field.value,
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
            **params
        )

        #保存先のファイル名を生成
        seed=params['seed']
        idx=params['idx']
        prompt=params['prompt']
        out_file = generate_output_filename(out_dir, idx, seed, prompt)
        out_dir=Path(out_dir)

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        res=encode_frames(res)

        save_images(res,out_dir.joinpath(now+f"{idx:02d}-{seed}"))
        save_animation(res,out_file)

        src_base64=image_to_animetion_base64(res,8)
        print(out_file.resolve().name)
        preview_img.src_base64=src_base64
        preview_img.update()

        pipeline = refresh_pipeline(pipeline)
        gc.collect()

    # Creating input widgets for each parameter with labels displaying the variable names and values
    prompt_field = ft.TextField(label="prompt",value=positive_prompt_default)
    n_prompt_field = ft.TextField(label="n_prompt",value=negative_prompt_default)
    seed_field = ft.TextField(label='seed', on_change=create_validator(int),keyboard_type=ft.KeyboardType.NUMBER)
    #steps_slider = ft.Slider(min=0, max=100, divisions=100,value=20, label="steps: {value}", on_change=create_validator(int))
    steps_slider= ft.TextField(label="steps",value=20,keyboard_type=ft.KeyboardType.NUMBER)
    guidance_scale_slider = ft.Slider(min=0.0, max=20.0, divisions=40,value=7.5, label="guidance_scale: {value}", on_change=create_validator(float))
    image_field = ft.TextField(label="image")
    strength_slider = ft.Slider(min=0.0, max=100.0, divisions=100,value=100.0 ,label="strength:{value}%", on_change=create_validator(float))
    canny_image_field = ft.TextField(label="controllnet_image",value=None)
    controlnet_preprocessing_field = ft.Dropdown(
        value="none",
        options=[
            ft.dropdown.Option("none"),
            ft.dropdown.Option("canny")
        ]
    )
    reference_image_field = ft.TextField(label="reference_image",value=None)
    image_guide_slider = ft.Slider(min=0.0, max=100., divisions=100,value=0.0,label="image_guide: {value}%", on_change=create_validator(float))
    schedule_dropdown = ft.Dropdown(
        value="continuous2",
        options=[
            ft.dropdown.Option("continuous2"),
            ft.dropdown.Option("continuous"),
            ft.dropdown.Option("uniform"),
            ft.dropdown.Option("continuous3"),
        ]
    )
    width_slider = ft.TextField(label="width",value=600,keyboard_type=ft.KeyboardType.NUMBER)
    height_slider = ft.TextField(label="height",value=800,keyboard_type=ft.KeyboardType.NUMBER)
    duration_slider = ft.Slider(min=16, max=512,divisions=62,label="duration: {value}",value=16, on_change=create_validator(int))
    out_dir_field = ft.TextField(label="out_dir",value="video_dir")
    controlnet_conditioning_scale_slider = ft.Slider(min=0.0, max=100.0, divisions=100, value=0.0,label="controlnet_conditioning_scale: {value}", on_change=create_validator(float))
    controlnet_conditioning_start_slider = ft.Slider(min=0.0, max=100.0, divisions=100, value=0.0,label="controlnet_conditioning_start: {value}", on_change=create_validator(float))
    controlnet_conditioning_end_slider   = ft.Slider(min=0.0, max=100.0, divisions=100, value=10.0,label="controlnet_conditioning_end: {value}", on_change=create_validator(float))
    controlnet_conditioning_bias_slider  = ft.Slider(min=0.0, max=100.0, divisions=100, value=0.0,label="controlnet_conditioning_bias: {value}", on_change=create_validator(float))

    # Creating the submit button
    submit_button = ft.ElevatedButton(text="Submit", on_click=button_clicked)

    preview_img = ft.Image(
        fit=ft.ImageFit.FIT_WIDTH,
        src=f'welcome.webp'
    )


    # Adding all the input widgets and the submit button to the page
    page.scroll=ft.ScrollMode.AUTO
    page.add(preview_img, submit_button,
        prompt_field, n_prompt_field, seed_field, steps_slider, guidance_scale_slider,
        image_field, strength_slider, canny_image_field,controlnet_preprocessing_field, reference_image_field, image_guide_slider,schedule_dropdown,
        width_slider, height_slider, duration_slider, out_dir_field,
        controlnet_conditioning_scale_slider, controlnet_conditioning_start_slider,
        controlnet_conditioning_end_slider, controlnet_conditioning_bias_slider
    )


# Running the application
ft.app(target=main,port=8550, view=ft.WEB_BROWSER,assets_dir="video_dir")


