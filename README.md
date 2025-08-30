# YOLOモデル テスト・評価フレームワーク

YOLOモデルのテスト、ONNX変換、量子化、性能評価を行う包括的なフレームワークです。

## プロジェクト概要

このプロジェクトは以下に焦点を当てています：
- YOLOモデル（v11とv12）のONNX形式への変換
- ONNXモデルをベースとデコーダーコンポーネントに分割し、推論を最適化
- 選択的演算子ターゲティングによるモデル量子化（INT8/INT16）
- mAP（mean Average Precision）メトリクスを使用した性能比較
- 分離されたモデルコンポーネントによる推論パイプライン

## プロジェクト構造

```
├── model_convert/              # モデル変換ユーティリティ
│   ├── yolo11n_nms.onnx       # NMS付きフルYOLO11nモデル
│   ├── yolo12n_nms.onnx       # NMS付きフルYOLO12nモデル
│   ├── yolo12_onnx_export.py  # YOLO12からONNXへの変換器
│   ├── yolo12_cut_onnx.py     # YOLO12モデル分割器
│   └── ...
├── yolo11_onnx_export.py      # YOLO11からONNXへの変換器
├── yolo11_cut_onnx.py         # YOLO11モデル分割器
├── yolo11_onnx_inference.py   # 分離モデル用推論エンジン
├── quantize_yolo11_base.py    # モデル量子化ツール
├── json_calculate_map.py      # mAP計算ユーティリティ
├── test.jpg                   # 推論用テスト画像
├── output.jpg                 # サンプル推論結果
├── *.json                     # JSON形式の検出結果
├── *.onnx                     # ONNXモデルファイル
├── CLAUDE.md                  # 開発ガイダンス
└── README.md                  # このファイル
```

## 主要機能

### 1. モデルエクスポートと分割
- YOLOのPyTorchモデルをNMS有効でONNX形式にエクスポート
- フルモデルをベース（特徴抽出）とデコーダー（NMS+検出）コンポーネントに分割
- 推論性能とメモリ使用量を最適化

### 2. モデル量子化
- キャリブレーションデータによる静的量子化
- INT8とINT16量子化のサポート
- 細調整された圧縮のための選択的演算子量子化
- 部分モデル量子化（特定レイヤー範囲）

### 3. 推論パイプライン
- 2段階推論：ベースモデル → デコーダーモデル
- 量子化モデルと元モデルのサポート
- 検出結果のJSON出力
- アノテーション付き画像生成

### 4. 性能評価
- IoUベースメトリクスを使用したmAP計算
- モデル比較機能
- 詳細なクラス別性能分析

## クイックスタート

### 前提条件
- Python 3.11+
- uvパッケージマネージャー

### インストール
```bash
# 依存関係をインストール
uv sync
```

### 基本的な使用方法

1. **YOLOモデルをONNXにエクスポート：**
```bash
uv run python yolo11_onnx_export.py
```

2. **モデルをベースとデコーダーに分割：**
```bash
uv run python yolo11_cut_onnx.py
```

3. **推論実行：**
```bash
uv run python yolo11_onnx_inference.py yolo11n_nms_base.onnx yolo11n_nms_decoder.onnx test.jpg
```

4. **モデル量子化：**
```bash
# INT8量子化
uv run python quantize_yolo11_base.py yolo11n_nms_base.onnx test.jpg

# 特定演算子でのINT16量子化
uv run python quantize_yolo11_base.py yolo11n_nms_base.onnx test.jpg -t int16 --ops Conv Gemm

# 部分モデル量子化
uv run python quantize_yolo11_base.py yolo11n_nms_base.onnx test.jpg --partial
```

5. **mAP計算：**
```bash
uv run python json_calculate_map.py ground_truth.json prediction.json -o results.txt
```

## 高度な機能

### モデル量子化オプション
- **フルモデル量子化：** サポートされている全演算子を量子化
- **選択的量子化：** 特定の演算子（Conv、Gemmなど）をターゲット
- **部分モデル量子化：** images → /model/model.23/Concat_*_output_0間の特定セクションを量子化
- **キャリブレーションベース：** 最適な量子化パラメータのために実データを使用

### 推論オプション
- **混合精度：** 量子化ベースモデルと元デコーダーを使用
- **カスタム出力パス：** 出力画像とJSONファイル名を指定
- **バッチ処理：** 複数画像の処理（拡張可能）

### 性能分析
- **クラス別AP：** 詳細なクラス別性能メトリクス
- **IoU閾値調整：** 調整可能な検出マッチング基準
- **モデル比較：** 並列性能評価

## ファイル形式詳細

### JSON検出形式
```json
{
  "image": "test.jpg",
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "score": 0.85,
      "class_id": 0
    }
  ]
}
```

### モデル命名規則
- オリジナル：`yolo11n_nms_base.onnx`
- 量子化：`yolo11n_nms_base_int8.onnx`
- 選択的量子化：`yolo11n_nms_base_int8_Conv_Gemm.onnx`
- 部分量子化：`yolo11n_nms_base_int8_partial.onnx`

## 技術詳細

### モデルアーキテクチャ
- **ベースモデル：** 特徴抽出と初期検出（`/model/model.21/Concat_5_output_0`で終了）
- **デコーダーモデル：** NMS処理と最終出力フォーマット
- **入力：** 640x640x3 RGB画像
- **出力：** NMS処理済み検出（Nx6：x1,y1,x2,y2,score,class_id）

### 量子化技術情報
- **INT8：** 互換性のためQOperator形式を使用
- **INT16：** QDQ形式を使用（16ビット必須）
- **キャリブレーション：** 単一画像キャリブレーション（複数画像に拡張可能）
- **演算子：** Conv、Gemm、MatMul、Add、Mulなどをサポート
- **部分量子化：** 特定レイヤー範囲（images入力から/model/model.23/Concat_*_output_0出力まで）

## 依存関係

主要パッケージ：
- **ultralytics：** YOLOモデル処理
- **onnxruntime：** ONNX推論と量子化
- **opencv-python：** 画像処理
- **onnx：** モデル操作と分析
- **numpy：** 数値計算
- **onnx-simplifier：** ONNXモデル最適化

## 開発

詳細な開発ガイダンスとコマンドリファレンスについては`CLAUDE.md`を参照してください。

## 性能結果

テストデータセットからのサンプル結果：
- **YOLO11n mAP：** 94.5%（YOLO12nベースライン対比）
- **YOLO12n mAP：** 100.0%（YOLO11nベースライン対比）
- **量子化サイズ削減：** 約71%（INT8）、精度損失は最小限
- **推論速度：** 分離モデルアーキテクチャにより向上

## 使用例

### 完全なワークフロー例
```bash
# 1. モデルをエクスポートして分割
uv run python yolo11_onnx_export.py
uv run python yolo11_cut_onnx.py

# 2. 元モデルで推論実行
uv run python yolo11_onnx_inference.py yolo11n_nms_base.onnx yolo11n_nms_decoder.onnx test.jpg

# 3. ベースモデルを量子化
uv run python quantize_yolo11_base.py yolo11n_nms_base.onnx test.jpg

# 4. 量子化モデルで推論実行
uv run python yolo11_onnx_inference.py yolo11n_nms_base_int8.onnx yolo11n_nms_decoder.onnx test.jpg

# 5. 性能比較
uv run python json_calculate_map.py test_yolo11n_nms_base.json test_yolo11n_nms_base_int8.json
```

## ライセンス

このプロジェクトは研究・教育目的で使用されます。