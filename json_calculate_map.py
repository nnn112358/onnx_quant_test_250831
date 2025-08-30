import json
import numpy as np
import argparse
import sys
from collections import defaultdict


def load_json_results(json_path):
    """JSONファイルから検出結果を読み込む"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['detections']
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {json_path}")
        return []
    except Exception as e:
        print(f"JSONファイルの読み込みエラー: {e}")
        return []


def calculate_iou(box1, box2):
    """2つのbounding boxのIoUを計算"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 交差領域の座標
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    # 交差領域の面積
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        intersection = 0
    else:
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # 各boxの面積
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # 結合領域の面積
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0
    
    return intersection / union


def calculate_ap_per_class(pred_detections, gt_detections, iou_threshold=0.5):
    """クラスごとのAverage Precision (AP)を計算"""
    if not pred_detections:
        return 0.0
    
    # 予測結果をスコアで降順ソート
    pred_detections = sorted(pred_detections, key=lambda x: x['score'], reverse=True)
    
    # GT boxesをセット
    gt_boxes = [det['bbox'] for det in gt_detections]
    gt_matched = [False] * len(gt_boxes)
    
    tp = []  # True Positives
    fp = []  # False Positives
    
    for pred in pred_detections:
        pred_box = pred['bbox']
        
        # GTとのマッチングを試行
        best_iou = 0
        best_gt_idx = -1
        
        for i, gt_box in enumerate(gt_boxes):
            if gt_matched[i]:
                continue
            
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        # IoU閾値を超え、まだマッチされていないGTがあればTP
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp.append(1)
            fp.append(0)
            gt_matched[best_gt_idx] = True
        else:
            tp.append(0)
            fp.append(1)
    
    # Precision-Recallカーブの計算
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / max(len(gt_boxes), 1)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    
    # APの計算（11点補間法）
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        p_interp = 0
        for r, p in zip(recalls, precisions):
            if r >= t:
                p_interp = max(p_interp, p)
        ap += p_interp / 11
    
    return ap


def calculate_map(detections1, detections2, model1_name, model2_name, output_file=None):
    """2つのモデルの検出結果からmAPを計算（一方をGTとして使用）"""
    results = []
    results.append(f"=== mAP計算: {model1_name} vs {model2_name} ===")
    results.append(f"{model2_name}をGround Truthとして使用")
    
    # クラスごとに分類
    classes1 = defaultdict(list)
    classes2 = defaultdict(list)
    
    for det in detections1:
        classes1[det['class_id']].append(det)
    
    for det in detections2:
        classes2[det['class_id']].append(det)
    
    # 共通のクラスを取得
    common_classes = set(classes1.keys()) & set(classes2.keys())
    
    if not common_classes:
        results.append("共通のクラスが見つかりません")
        if output_file:
            print("\n".join(results), file=output_file)
        else:
            for line in results:
                print(line)
        return 0.0
    
    results.append(f"共通クラス数: {len(common_classes)}")
    results.append(f"共通クラス: {sorted(common_classes)}")
    
    # クラスごとのAPを計算
    aps = []
    for class_id in sorted(common_classes):
        pred_dets = classes1[class_id]
        gt_dets = classes2[class_id]
        
        ap = calculate_ap_per_class(pred_dets, gt_dets)
        aps.append(ap)
        
        results.append(f"クラス {class_id}: AP = {ap:.3f} (予測:{len(pred_dets)}, GT:{len(gt_dets)})")
    
    # mAPを計算
    mean_ap = np.mean(aps)
    results.append(f"\n{model1_name} mAP: {mean_ap:.3f}")
    
    # 結果を出力
    if output_file:
        print("\n".join(results), file=output_file)
    else:
        for line in results:
            print(line)
    
    return mean_ap


def main():
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description="YOLO検出結果のmAP計算")
    parser.add_argument("json1", help="1つ目のJSONファイル")
    parser.add_argument("json2", help="2つ目のJSONファイル")
    parser.add_argument("-o", "--output", help="出力テキストファイル", default=None)
    
    args = parser.parse_args()
    
    # JSONファイルのパス
    json1_path = args.json1
    json2_path = args.json2
    
    # 出力ファイルの設定
    output_file = None
    if args.output:
        output_file = open(args.output, 'w', encoding='utf-8')
    
    try:
        # 初期情報を出力
        header = [
            "=== mAP計算スクリプト ===",
            f"モデル1: {json1_path}",
            f"モデル2: {json2_path}"
        ]
        
        if output_file:
            for line in header:
                print(line, file=output_file)
        else:
            for line in header:
                print(line)
        
        # 検出結果を読み込み
        detections1 = load_json_results(json1_path)
        detections2 = load_json_results(json2_path)
        
        if not detections1:
            error_msg = f"エラー: {json1_path} の読み込みに失敗しました"
            if output_file:
                print(error_msg, file=output_file)
            else:
                print(error_msg)
            return
        
        if not detections2:
            error_msg = f"エラー: {json2_path} の読み込みに失敗しました"
            if output_file:
                print(error_msg, file=output_file)
            else:
                print(error_msg)
            return
        
        # 検出数情報
        model1_name = json1_path.split('_')[1] if '_' in json1_path else "Model1"
        model2_name = json2_path.split('_')[1] if '_' in json2_path else "Model2"
        
        detection_info = [
            f"\n検出結果:",
            f"{model1_name}: {len(detections1)} 個のオブジェクト",
            f"{model2_name}: {len(detections2)} 個のオブジェクト"
        ]
        
        if output_file:
            for line in detection_info:
                print(line, file=output_file)
        else:
            for line in detection_info:
                print(line)
        
        # json1を真値としてmAPを計算
        separator = f"\n{'='*50}"
        if output_file:
            print(separator, file=output_file)
        else:
            print(separator)
        
        # json1を真値、json2を予測として計算
        map2 = calculate_map(detections2, detections1, model2_name, model1_name, output_file)
        
        # 最終結果
        final_results = [
            f"\n{'='*50}",
            "=== 最終結果 ===",
            f"{model1_name}を真値として使用",
            f"{model2_name} mAP: {map2:.3f}"
        ]
        
        if output_file:
            for line in final_results:
                print(line, file=output_file)
        else:
            for line in final_results:
                print(line)
    
    finally:
        if output_file:
            output_file.close()
            print(f"結果を保存しました: {args.output}")


if __name__ == "__main__":
    main()