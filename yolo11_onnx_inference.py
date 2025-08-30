import onnxruntime as ort
import cv2
import numpy as np
import json
import os
import argparse


def load_models(base_model_path, decoder_model_path):
    """ベースモデルとデコーダーモデルをロードする関数"""
    try:
        providers = ['CPUExecutionProvider']
        base_session = ort.InferenceSession(base_model_path, providers=providers)
        decoder_session = ort.InferenceSession(decoder_model_path, providers=providers)
        print(f"ベースモデルを正常にロードしました: {base_model_path}")
        print(f"デコーダーモデルを正常にロードしました: {decoder_model_path}")
        return base_session, decoder_session
    except Exception as e:
        print(f"モデルのロード中にエラーが発生しました: {e}")
        return None, None


def prepare_image(image_path, input_size=640):
    """画像を前処理する関数"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"画像の読み込みに失敗しました: {image_path}")
        return None, None
    
    original_height, original_width = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (input_size, input_size))
    
    input_data = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    
    return input_data, (original_width, original_height, img)


def run_inference(base_session, decoder_session, input_data):
    """ベースモデルとデコーダーモデルで推論を実行する関数"""
    try:
        # ベースモデルで推論
        base_input_name = base_session.get_inputs()[0].name
        base_outputs = base_session.run(None, {base_input_name: input_data})
        
        # デコーダーモデルの入力名を取得
        decoder_input_names = [input.name for input in decoder_session.get_inputs()]
        
        # ベースモデルの出力数とデコーダーモデルの入力数を確認
        print(f"ベースモデル出力数: {len(base_outputs)}")
        print(f"デコーダーモデル入力数: {len(decoder_input_names)}")
        print(f"デコーダーモデル入力名: {decoder_input_names}")
        
        # ベースモデルの出力をデコーダーモデルの入力として使用
        if len(base_outputs) == len(decoder_input_names):
            # 出力数と入力数が一致する場合
            decoder_input_dict = {name: output for name, output in zip(decoder_input_names, base_outputs)}
        elif len(decoder_input_names) == 1:
            # デコーダーが1つの入力を期待する場合（従来の動作）
            decoder_input_dict = {decoder_input_names[0]: base_outputs[0]}
        else:
            raise ValueError(f"ベースモデル出力数({len(base_outputs)})とデコーダーモデル入力数({len(decoder_input_names)})が一致しません")
        
        decoder_outputs = decoder_session.run(None, decoder_input_dict)
        
        return decoder_outputs
    except Exception as e:
        print(f"推論中にエラーが発生しました: {e}")
        return None


def process_detections(outputs, original_size, input_size=640, conf_threshold=0.25):
    """検出結果を処理してbboxを出力する関数 (NMS適用済みモデル用)"""
    original_width, original_height, original_img = original_size
    
    detections = outputs[0]
    print(f"検出形状: {detections.shape}")
    
    # NMS適用済みモデルの出力形式: (1, 300, 6) -> (300, 6)
    # 各検出結果: [x1, y1, x2, y2, confidence, class_id]
    if len(detections.shape) == 3:
        detections = detections[0]  # (1, 300, 6) -> (300, 6)
    
    # 信頼度による有効な検出のフィルタリング
    # confidence > 0 の検出結果のみを使用
    valid_mask = detections[:, 4] > 0  # confidenceが0より大きい
    valid_detections = detections[valid_mask]
    
    # さらに信頼度閾値でフィルタリング
    conf_mask = valid_detections[:, 4] > conf_threshold
    filtered_detections = valid_detections[conf_mask]
    
    print(f"有効な検出数: {len(filtered_detections)}")
    
    # スケーリング係数 (入力画像サイズから元画像サイズへ)
    scale_x = original_width / input_size
    scale_y = original_height / input_size
    
    bboxes = []
    result_img = original_img.copy()
    
    # クラスごとに色を設定
    np.random.seed(42)
    colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) 
              for _ in range(80)]
    
    for detection in filtered_detections:
        # NMS済み出力: [x1, y1, x2, y2, confidence, class_id]
        x1, y1, x2, y2, confidence, class_id = detection
        
        # 座標を元画像サイズにスケーリング
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
        
        # 座標を画像範囲内に調整
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(original_width - 1, x2)
        y2 = min(original_height - 1, y2)
        
        class_id = int(class_id)
        
        # bboxデータを保存
        bbox_data = {
            'bbox': [x1, y1, x2, y2],
            'score': float(confidence),
            'class_id': class_id
        }
        bboxes.append(bbox_data)
        
        # 画像に描画
        color = colors[class_id % len(colors)]
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        
        label = f"Class {class_id}: {confidence:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        y1_label = max(0, y1 - label_height - 10)
        cv2.rectangle(result_img, (x1, y1_label), (x1 + label_width, y1), color, -1)
        cv2.putText(result_img, label, (x1, y1 - 7 if y1 > label_height + 10 else y1 + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return result_img, bboxes


def save_results_as_json(bboxes, image_path, base_model_path):
    """検出結果をJSON形式で保存する関数"""
    # 画像ファイル名とONNXモデル名からJSONファイル名を生成
    image_base_name = os.path.splitext(os.path.basename(image_path))[0]
    model_base_name = os.path.splitext(os.path.basename(base_model_path))[0]
    json_path = f"{image_base_name}_{model_base_name}.json"
    
    # JSON出力用のデータを準備
    results = {
        "image": os.path.basename(image_path),
        "detections": []
    }
    
    for bbox in bboxes:
        detection = {
            "bbox": bbox['bbox'],
            "score": bbox['score'],
            "class_id": bbox['class_id']
        }
        results["detections"].append(detection)
    
    # JSONファイルに保存
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"検出結果をJSONで保存しました: {json_path}")
    return json_path


def main():
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description="YOLO ONNX推論（分離モデル版）")
    parser.add_argument("base_model", help="ベースモデルのONNXファイル")
    parser.add_argument("decoder_model", help="デコーダーモデルのONNXファイル")
    parser.add_argument("image", help="入力画像ファイル")
    parser.add_argument("-o", "--output", help="出力画像ファイル", default="output.jpg")
    
    args = parser.parse_args()
    
    base_model_path = args.base_model
    decoder_model_path = args.decoder_model
    image_path = args.image
    output_path = args.output
    
    print(f"=== YOLO ONNX推論 (分離モデル版) ===")
    print(f"ベースモデル: {base_model_path}")
    print(f"デコーダーモデル: {decoder_model_path}")
    print(f"画像: {image_path}")
    
    # ファイルの存在確認
    if not os.path.exists(base_model_path):
        print(f"エラー: ベースモデルが見つかりません: {base_model_path}")
        return
    
    if not os.path.exists(decoder_model_path):
        print(f"エラー: デコーダーモデルが見つかりません: {decoder_model_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"エラー: 画像ファイルが見つかりません: {image_path}")
        return
    
    # モデルをロード
    base_session, decoder_session = load_models(base_model_path, decoder_model_path)
    if base_session is None or decoder_session is None:
        return
    
    # 画像を準備
    input_data, original_size = prepare_image(image_path)
    if input_data is None:
        return
    
    # 推論を実行
    print("推論実行中...")
    outputs = run_inference(base_session, decoder_session, input_data)
    if outputs is None:
        return
    
    # 検出結果を処理
    result_img, bboxes = process_detections(outputs, original_size)
    
    # 結果を表示
    print(f"\n=== 検出結果 ===")
    print(f"検出されたオブジェクト数: {len(bboxes)}")
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox['bbox']
        print(f"  {i+1}: クラス{bbox['class_id']}, スコア{bbox['score']:.3f}, "
              f"bbox[{x1}, {y1}, {x2}, {y2}]")
    
    # 結果画像を保存
    cv2.imwrite(output_path, result_img)
    print(f"\n結果画像を保存しました: {output_path}")
    
    # 結果をJSONで保存
    save_results_as_json(bboxes, image_path, base_model_path)


if __name__ == "__main__":
    main()