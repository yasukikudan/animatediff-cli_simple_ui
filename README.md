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
python -m pip install torch torchvision torchaudio flet
python -m pip install -e '.[dev]'
python -m pip install -e '.[rife]'

```




