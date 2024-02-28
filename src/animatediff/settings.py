import json
import logging
from functools import lru_cache
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from pydantic import BaseConfig, BaseSettings, Field
from pydantic.env_settings import (EnvSettingsSource, InitSettingsSource,
                                   SecretsSettingsSource,
                                   SettingsSourceCallable)

from animatediff import get_dir
from animatediff.schedulers import DiffusionScheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CKPT_EXTENSIONS = [".pt", ".ckpt", ".pth", ".safetensors"]


class JsonSettingsSource:
    __slots__ = ["json_config_path"]

    def __init__(
        self,
        json_config_path: Optional[Union[PathLike, list[PathLike]]] = list(),
    ) -> None:
        if isinstance(json_config_path, list):
            self.json_config_path = [Path(path) for path in json_config_path]
        else:
            self.json_config_path = [Path(json_config_path)] if json_config_path is not None else []

    def __call__(self, settings: BaseSettings) -> Dict[str, Any]:  # noqa C901
        classname = settings.__class__.__name__
        encoding = settings.__config__.env_file_encoding
        if len(self.json_config_path) == 0:
            pass  # no json config provided

        merged_config = dict()  # create an empty dict to merge configs into
        for idx, path in enumerate(self.json_config_path):
            if path.exists() and path.is_file():  # check if the path exists and is a file
                logger.debug(f"{classname}: loading config #{idx+1} from {path}")
                merged_config.update(json.loads(path.read_text(encoding=encoding)))
                logger.debug(f"{classname}: config state #{idx+1}: {merged_config}")
            else:
                raise FileNotFoundError(f"{classname}: config #{idx+1} at {path} not found or not a file")

        logger.debug(f"{classname}: loaded config: {merged_config}")
        return merged_config  # return the merged config

    def __repr__(self) -> str:
        return f"JsonSettingsSource(json_config_path={repr(self.json_config_path)})"


class JsonConfig(BaseConfig):
    json_config_path: Optional[Union[Path, list[Path]]] = None
    env_file_encoding: str = "utf-8"

    @classmethod
    def customise_sources(
        cls,
        init_settings: InitSettingsSource,
        env_settings: EnvSettingsSource,
        file_secret_settings: SecretsSettingsSource,
    ) -> Tuple[SettingsSourceCallable, ...]:
        # pull json_config_path from init_settings if passed, otherwise use the class var
        json_config_path = init_settings.init_kwargs.pop("json_config_path", cls.json_config_path)

        logger.debug(f"Using JsonSettingsSource for {cls.__name__}")
        json_settings = JsonSettingsSource(json_config_path=json_config_path)

        # return the new settings sources
        return (
            init_settings,
            json_settings,
        )


class InferenceConfig(BaseSettings):
    unet_additional_kwargs: dict[str, Any]
    noise_scheduler_kwargs: dict[str, Any]

    class Config(JsonConfig):
        json_config_path: Path


@lru_cache(maxsize=2)
def get_infer_config(
    config_path: Path = get_dir("config").joinpath("inference/default.json"),
) -> InferenceConfig:
    settings = InferenceConfig(json_config_path=config_path)
    return settings


def get_pipeline_config(
        config_path: Path = get_dir("config").joinpath("pipeline/default.json"),
):
    # 設定ファイルの読み込み
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
        return config



class ModelConfig(BaseSettings):
    name: str = Field(...)  # Config name, not actually used for much of anything
    base: Optional[Path] = Field(None)  # Path to base checkpoint (if using a LoRA)
    path: Path = Field(...)  # Path to the model or LoRA checkpoint
    motion_module: Path = Field(...)  # Path to the motion module
    compile: bool = Field(False)  # whether to compile the model with TorchDynamo
    seed: list[int] = Field([])  # Seed(s) for the random number generators
    scheduler: DiffusionScheduler = Field(DiffusionScheduler.k_dpmpp_2m)  # Scheduler to use
    steps: int = 25  # Number of inference steps to run
    guidance_scale: float = 7.5  # CFG scale to use
    clip_skip: int = 1  # skip the last N-1 layers of the CLIP text encoder
    prompt: list[str] = Field([])  # Prompt(s) to use
    n_prompt: list[str] = Field([])  # Anti-prompt(s) to use

    controlnet: Optional[Path] = Field(None)  # Path to the controlnet checkpoint


    # Directly specifying the path to image files
    image: Optional[Union[Path, str]] = Field(None, description="Path to the image file")
    canny_image: Optional[Union[Path, str]] = Field(None, description="Path to the canny-processed image file")
    reference_image: Optional[Union[Path, str]] = Field(None, description="Path to the reference image file")
    strength: Optional[float] = Field(1.0, description="Strength indicating the importance of the image", gte=0, le=1)
    image_guide: Optional[float]    = Field(1.0, description="Image_guide indicating the importance of the image", gte=0, le=1)

    controlnet_conditioning_scale:Optional[float]= Field(0.01, description="A coefficient indicating the scale of the control net conditioning.", gte=0, le=1)
    controlnet_conditioning_start:Optional[float]= Field(0.1, description="The intensity at which the control net conditioning starts.", gte=0, le=2)
    controlnet_conditioning_end  :Optional[float]= Field(0.2, description="The intensity at which the control net conditioning ends.", gte=0, le=1)
    controlnet_conditioning_bias :Optional[float]= Field(1.0, description="A coefficient indicating the bias of the control net conditioning.", gte=0, le=1)
    controlnet_preprocessing :str = Field("none", description="The preprocessing method for the control net conditioning.", regex="^(none|canny|gray)$")

    # Specifying the folder where the files are stored
    #image_folder: Optional[Path] = Field(None, description="Folder containing the image files")
    #canny_image_folder: Optional[Path] = Field(None, description="Folder containing the canny-processed image files")
    #reference_image_folder: Optional[Path] = Field(None, description="Folder containing the reference image files")

    # Strengths and weights indicating the importance of each image
    #strength: Optional[float] = Field(1.0, description="Strength indicating the importance of the image", gt=0, le=1)
    #image_guide: Optional[float]    = Field(1.0, description="Image_guide indicating the importance of the image", gt=0, le=1)

    #canny_image_weight: Optional[float] = Field(1.0, description="Weight indicating the importance of the canny image", gt=0, le=2)
    #reference_image_weight: Optional[float] = Field(1.0, description="Weight indicating the importance of the reference image", gt=0, le=2)

    class Config(JsonConfig):
        json_config_path: Path

    @property
    def save_name(self):
        if self.base is not None and str(self.base) != ".":
            return f"{self.name.lower()}-{self.path.stem.lower()}-{self.base.stem.lower()}"
        else:
            return f"{self.name.lower()}-{self.path.stem.lower()}"


@lru_cache(maxsize=2)
def get_model_config(config_path: Path) -> ModelConfig:
    settings = ModelConfig(json_config_path=config_path)
    return settings
