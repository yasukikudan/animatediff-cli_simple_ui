from os import PathLike
from pathlib import Path

import torch
from einops import rearrange
from PIL import Image
from torch import Tensor
from torchvision.utils import save_image
from tqdm.rich import tqdm
from torchvision.transforms.functional import to_pil_image

def encode_frames(video: Tensor) -> [Image]:
    # ビデオテンソルをフレームのリストに再構成
    frames = rearrange(video, "b c t h w -> t b c h w").squeeze(1)
    # PIL Imageオブジェクトのリストを作成
    images = [to_pil_image(frame) for frame in frames]
    return images

def save_images(flames:Image, frames_dir: PathLike):
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(tqdm(flames, desc=f"Saving frames to {frames_dir.stem}")):
        frame.save(frames_dir.joinpath(f"{idx:04d}.png"))
        #save_image(frame, frames_dir.joinpath(f"{idx:04d}.png"))

def save_animation(flames:Image, save_path: PathLike, fps: int = 8):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    images = [frame for frame in flames]
    images[0].save(
        fp=save_path, format="WEBP", append_images=images[1:], save_all=True, duration=(1 / fps * 1000), loop=0
    )

def save_frames(video: Tensor, frames_dir: PathLike):
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames = rearrange(video, "b c t h w -> t b c h w")
    for idx, frame in enumerate(tqdm(frames, desc=f"Saving frames to {frames_dir.stem}")):
        save_image(frame, frames_dir.joinpath(f"{idx:04d}.png"))


def save_video(video: Tensor, save_path: PathLike, fps: int = 8):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if video.ndim == 5:
        # batch, channels, frame, width, height -> frame, channels, width, height
        frames = video.permute(0, 2, 1, 3, 4).squeeze(0)
    elif video.ndim == 4:
        # channels, frame, width, height -> frame, channels, width, height
        frames = video.permute(1, 0, 2, 3)
    else:
        raise ValueError(f"video must be 4 or 5 dimensional, got {video.ndim}")

    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    frames = frames.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()

    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        fp=save_path, format="GIF", append_images=images[1:], save_all=True, duration=(1 / fps * 1000), loop=0
    )


def path_from_cwd(path: PathLike) -> str:
    path = Path(path)
    return str(path.absolute().relative_to(Path.cwd()))
