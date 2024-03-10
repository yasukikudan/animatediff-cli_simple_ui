# animatediff-cli_simple_ui

`animatediff-cli_simple_ui`は、[animatediff-cli](https://github.com/neggles/animatediff-cli)のためのシンプルなGUI実装です。これは、Pythonの[flet](https://flet.dev/)ライブラリを使用しており、より直感的な操作を実現しています。特に、webuiのanimatediffに比べてメモリ使用量を大幅に削減しており、10GBのVRAMを搭載した環境で800x600の解像度、およびコントロールネットを使用した100フレーム以上の動画生成を可能としています。

## 特徴

- **省メモリ実装**: 10GBのVRAMで高解像度かつ長時間の動画生成を実現。
- **シンプルなGUI**: fletライブラリにより、GUIでのパラメータ変更が可能
- **高度な動画生成**: コントロールネットを使用した出力画像の調整が可能

## アニメーションサンプル
![サンプル](https://raw.githubusercontent.com/yasukikudan/animatediff-cli_simple_ui/main/generate_anime_sample.webp)

## 画像構成
![サンプル](https://raw.githubusercontent.com/yasukikudan/animatediff-cli_simple_ui/main/simpe_ui_image.png)

## 使い方

### 環境構築

```sh
git clone https://github.com/yasukikudan/animatediff-cli_simple_ui
cd animatediff-cli
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision torchaudio flet opencv-python
python -m pip install -e '.[dev]'
python -m pip install -e '.[rife]'
mkdir video_dir
```


### モデルデータの配置場所
config/pipeline/default.json
下記パスにモデルファイルを配置
```json
{
"base_model": "data/models/huggingface/stable-diffusion-v1-5/",
"controlnet_model": "data/models/controlnet/control_v11e_sd15_ip2p",
"checkpoint_path": "data/models/sd/SDHK04.safetensors",
"motion_module_path": "data/models/motion-module/mm_sd_v14.safetensors"
}
```


### 複数枚の画像からアニメーションを作成する方法


#rifeを使って中間フレームを作成 テスト画像は4枚なので4x8で32枚の画像が生成される
```sh
animatediff rife interpolate   -M 8 --in-fps 2 --out-fps 8 src/animatediff/image_dir/
#中間フレームの出力先はsrc/animatediff/image_dir-rife/
```

GUI起動
```sh
python src/animatediff/gui.py
```

imageの項目に中間フレームの出力先を指定 img2imgの強度を50〜60程度に指定して実行する






