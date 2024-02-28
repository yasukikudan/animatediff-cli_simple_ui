# animatediff-cli_simple_ui

`animatediff-cli_simple_ui`は、[animatediff-cli](https://github.com/neggles/animatediff-cli)のためのシンプルなGUI実装です。これは、Pythonの[flet](https://flet.dev/)ライブラリを使用しており、より直感的な操作を実現しています。特に、webuiのanimatediffに比べてメモリ使用量を大幅に削減しており、10GBのVRAMを搭載した環境で800x600の解像度、およびコントロールネットを使用した100フレーム以上の動画生成を可能としています。

## 特徴

- **省メモリ実装**: 10GBのVRAMで高解像度かつ長時間の動画生成を実現。
- **シンプルなGUI**: fletライブラリにより、直感的な操作性を提供。
- **高度な動画生成**: コントロールネットを使用した詳細な動画制御が可能。
