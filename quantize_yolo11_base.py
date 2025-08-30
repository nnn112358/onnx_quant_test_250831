import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
import cv2
import numpy as np
import os
import argparse


class YOLOCalibrationDataReader(CalibrationDataReader):
    """YOLO用のキャリブレーションデータリーダー"""
    
    def __init__(self, calibration_image_path, input_size=640):
        self.calibration_image_path = calibration_image_path
        self.input_size = input_size
        self.data_processed = False
        
    def get_next(self):
        """次のキャリブレーションデータを取得"""
        if self.data_processed:
            return None
        
        # 画像を読み込み・前処理
        img = cv2.imread(self.calibration_image_path)
        if img is None:
            print(f"画像の読み込みに失敗しました: {self.calibration_image_path}")
            return None
        
        # 前処理（yolo11_onnx_inference.pyのprepare_image関数と同じ）
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_size, self.input_size))
        
        input_data = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        
        self.data_processed = True
        
        return {"images": input_data}


def get_model_operators(model_path):
    """モデル内の演算子一覧を取得"""
    try:
        model = onnx.load(model_path)
        operators = set()
        for node in model.graph.node:
            operators.add(node.op_type)
        return sorted(list(operators))
    except Exception as e:
        print(f"モデル解析エラー: {e}")
        return []




def quantize_yolo_base_model(input_model_path, output_model_path, calibration_image_path, 
                           quantization_type="int8", operators_to_quantize=None):
    """YOLO11ベースモデルを静的量子化"""
    
    # 量子化タイプの設定
    if quantization_type.lower() == "int8":
        quant_format = QuantFormat.QOperator
        activation_type = QuantType.QUInt8
        weight_type = QuantType.QInt8
        type_label = "INT8"
    elif quantization_type.lower() == "int16":
        quant_format = QuantFormat.QDQ  # INT16にはQDQフォーマットが必要
        activation_type = QuantType.QUInt16
        weight_type = QuantType.QInt16
        type_label = "INT16"
    else:
        print(f"エラー: サポートされていない量子化タイプ: {quantization_type}")
        return False
    
    print(f"=== YOLO11 Base Model {type_label}量子化 ===")
    print(f"入力モデル: {input_model_path}")
    print(f"出力モデル: {output_model_path}")
    print(f"キャリブレーション画像: {calibration_image_path}")
    
    # 入力ファイルの存在確認
    if not os.path.exists(input_model_path):
        print(f"エラー: 入力モデルが見つかりません: {input_model_path}")
        return False
    
    if not os.path.exists(calibration_image_path):
        print(f"エラー: キャリブレーション画像が見つかりません: {calibration_image_path}")
        return False
    
    # 演算子情報を表示
    if operators_to_quantize:
        print(f"量子化対象演算子: {operators_to_quantize}")
    else:
        print("量子化対象: 全演算子")
    
    try:
        # キャリブレーションデータリーダーを作成
        calibration_data_reader = YOLOCalibrationDataReader(calibration_image_path)
        
        print(f"{type_label}量子化を開始します...")
        
        # 静的量子化を実行
        quantize_static(
            model_input=input_model_path,
            model_output=output_model_path,
            calibration_data_reader=calibration_data_reader,
            quant_format=quant_format,
            activation_type=activation_type,
            weight_type=weight_type,
            op_types_to_quantize=operators_to_quantize,  # 特定の演算子のみ量子化
        )
        
        print(f"{type_label}量子化が完了しました: {output_model_path}")
        
        # ファイルサイズを比較
        original_size = os.path.getsize(input_model_path)
        quantized_size = os.path.getsize(output_model_path)
        reduction = (1 - quantized_size / original_size) * 100
        
        print(f"\nファイルサイズ比較:")
        print(f"元モデル: {original_size / 1024 / 1024:.2f} MB")
        print(f"{type_label}量子化モデル: {quantized_size / 1024 / 1024:.2f} MB")
        print(f"サイズ削減率: {reduction:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"{type_label}量子化中にエラーが発生しました: {e}")
        return False
    


def verify_quantized_model(quantized_model_path, test_image_path):
    """量子化モデルの動作確認"""
    print(f"\n=== 量子化モデルの動作確認 ===")
    
    try:
        # モデルをロード
        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(quantized_model_path, providers=providers)
        print(f"量子化モデルを正常にロードしました: {quantized_model_path}")
        
        # テスト画像を準備
        img = cv2.imread(test_image_path)
        if img is None:
            print(f"テスト画像の読み込みに失敗しました: {test_image_path}")
            return False
        
        # 前処理
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 640))
        input_data = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        
        # 推論実行
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_data})
        
        print(f"推論成功! 出力形状: {outputs[0].shape}")
        return True
        
    except Exception as e:
        print(f"動作確認中にエラーが発生しました: {e}")
        return False


def main():
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description="YOLO11ベースモデルの量子化")
    parser.add_argument("input_model", help="入力ONNXモデル")
    parser.add_argument("calibration_image", help="キャリブレーション用画像")
    parser.add_argument("-t", "--type", choices=["int8", "int16"], default="int8", 
                       help="量子化タイプ (int8 または int16, デフォルト: int8)")
    parser.add_argument("-o", "--output", help="出力モデル名 (未指定時は自動生成)")
    parser.add_argument("--list-ops", action="store_true", help="モデル内の演算子一覧を表示")
    parser.add_argument("--ops", nargs="*", help="量子化対象の演算子を指定 (例: --ops Conv Gemm)")
    
    args = parser.parse_args()
    
    # 演算子一覧表示モード
    if args.list_ops:
        print(f"=== {args.input_model} 内の演算子一覧 ===")
        operators = get_model_operators(args.input_model)
        if operators:
            for i, op in enumerate(operators, 1):
                print(f"{i:2d}. {op}")
        else:
            print("演算子の取得に失敗しました")
        return
    
    # 出力ファイル名の設定
    if args.output:
        output_model_path = args.output
    else:
        # 自動生成: 元ファイル名 + _int8 または _int16
        base_name = os.path.splitext(args.input_model)[0]
        ext = os.path.splitext(args.input_model)[1]
        suffix = f"_{args.type.lower()}"
        if args.ops:
            suffix += f"_{'_'.join(args.ops)}"
        output_model_path = f"{base_name}{suffix}{ext}"
    
    print(f"入力モデル: {args.input_model}")
    print(f"出力モデル: {output_model_path}")
    print(f"量子化タイプ: {args.type.upper()}")
    print(f"キャリブレーション画像: {args.calibration_image}")
    
    # 演算子の検証
    if args.ops:
        available_ops = get_model_operators(args.input_model)
        invalid_ops = [op for op in args.ops if op not in available_ops]
        if invalid_ops:
            print(f"警告: 以下の演算子はモデルに存在しません: {invalid_ops}")
            print(f"利用可能な演算子: {available_ops}")
    
    # 量子化を実行
    success = quantize_yolo_base_model(
        args.input_model, 
        output_model_path, 
        args.calibration_image,
        args.type,
        args.ops  # 特定の演算子のみ量子化
    )
    
    if success:
        # 動作確認
        verify_quantized_model(output_model_path, args.calibration_image)
    else:
        print("量子化に失敗しました")


if __name__ == "__main__":
    main()