#!/usr/bin/env python3
"""
mAP (mean Average Precision) calculation script for YOLO detection results.
Compares prediction JSON files against ground truth JSON file.
"""

import json
import numpy as np
import argparse
import os
from typing import List, Dict, Tuple


def load_detections(json_path: str) -> List[Dict]:
    """JSONファイルから検出結果を読み込む"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['detections']


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """2つのbounding boxのIoU (Intersection over Union)を計算する"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 交差領域の座標を計算
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    # 交差領域の面積を計算
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        intersection = 0.0
    else:
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # 各ボックスの面積を計算
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # 和集合の面積を計算
    union = area1 + area2 - intersection
    
    # IoUを計算
    if union == 0:
        return 0.0
    return intersection / union


def calculate_ap_for_class(gt_detections: List[Dict], pred_detections: List[Dict], 
                          class_id: int, iou_threshold: float = 0.5) -> float:
    """特定のクラスに対するAP (Average Precision)を計算する"""
    
    # 該当クラスの検出結果を抽出
    gt_class = [det for det in gt_detections if det['class_id'] == class_id]
    pred_class = [det for det in pred_detections if det['class_id'] == class_id]
    
    if len(gt_class) == 0:
        return 0.0 if len(pred_class) > 0 else 1.0
    
    if len(pred_class) == 0:
        return 0.0
    
    # 予測結果をスコアでソート（降順）
    pred_class = sorted(pred_class, key=lambda x: x['score'], reverse=True)
    
    # TP, FPを計算
    tp = np.zeros(len(pred_class))
    fp = np.zeros(len(pred_class))
    gt_matched = [False] * len(gt_class)
    
    for pred_idx, pred in enumerate(pred_class):
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(gt_class):
            if gt_matched[gt_idx]:
                continue
                
            iou = calculate_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[pred_idx] = 1
            gt_matched[best_gt_idx] = True
        else:
            fp[pred_idx] = 1
    
    # 累積TPとFPを計算
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # PrecisionとRecallを計算
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    recall = tp_cumsum / len(gt_class)
    
    # APを計算（11点補間法）
    ap = 0.0
    for threshold in np.linspace(0, 1, 11):
        precision_at_recall = 0.0
        for i in range(len(recall)):
            if recall[i] >= threshold:
                precision_at_recall = max(precision_at_recall, precision[i])
        ap += precision_at_recall / 11
    
    return ap


def calculate_map(gt_json_path: str, pred_json_path: str, iou_threshold: float = 0.5) -> Tuple[float, Dict[int, float]]:
    """mAPを計算する"""
    
    # 検出結果を読み込み
    gt_detections = load_detections(gt_json_path)
    pred_detections = load_detections(pred_json_path)
    
    # 全クラスIDを取得
    gt_classes = set(det['class_id'] for det in gt_detections)
    pred_classes = set(det['class_id'] for det in pred_detections)
    all_classes = gt_classes.union(pred_classes)
    
    # 各クラスのAPを計算
    class_aps = {}
    for class_id in all_classes:
        ap = calculate_ap_for_class(gt_detections, pred_detections, class_id, iou_threshold)
        class_aps[class_id] = ap
    
    # mAPを計算
    if len(class_aps) == 0:
        map_score = 0.0
    else:
        map_score = sum(class_aps.values()) / len(class_aps)
    
    return map_score, class_aps


def main():
    parser = argparse.ArgumentParser(description="Calculate mAP between ground truth and prediction JSON files")
    parser.add_argument("ground_truth", help="Ground truth JSON file")
    parser.add_argument("prediction", help="Prediction JSON file")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="IoU threshold (default: 0.5)")
    parser.add_argument("-o", "--output", help="Output results to text file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.ground_truth):
        print(f"エラー: Ground truthファイルが見つかりません: {args.ground_truth}")
        return
    
    if not os.path.exists(args.prediction):
        print(f"エラー: Predictionファイルが見つかりません: {args.prediction}")
        return
    
    print(f"=== mAP計算 ===")
    print(f"Ground Truth: {args.ground_truth}")
    print(f"Prediction: {args.prediction}")
    print(f"IoU閾値: {args.threshold}")
    print()
    
    # mAPを計算
    map_score, class_aps = calculate_map(args.ground_truth, args.prediction, args.threshold)
    
    # 結果を表示
    print("=== クラス別AP ===")
    sorted_classes = sorted(class_aps.keys())
    for class_id in sorted_classes:
        ap = class_aps[class_id]
        print(f"クラス {class_id}: AP = {ap:.16f}")
    
    print(f"\n=== 全体結果 ===")
    print(f"mAP@{args.threshold}: {map_score:.16f}")
    
    # ファイル出力
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(f"=== mAP計算結果 ===\n")
            f.write(f"Ground Truth: {args.ground_truth}\n")
            f.write(f"Prediction: {args.prediction}\n")
            f.write(f"IoU閾値: {args.threshold}\n\n")
            
            f.write("=== クラス別AP ===\n")
            for class_id in sorted_classes:
                ap = class_aps[class_id]
                f.write(f"クラス {class_id}: AP = {ap:.16f}\n")
            
            f.write(f"\n=== 全体結果 ===\n")
            f.write(f"mAP@{args.threshold}: {map_score:.16f}\n")
        
        print(f"\n結果をファイルに保存しました: {args.output}")


if __name__ == "__main__":
    main()