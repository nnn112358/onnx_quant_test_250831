#!/usr/bin/env python3

import onnx
import sys
from collections import Counter

def count_operators(model_path):
    """Count the number of each operator type in an ONNX model"""
    model = onnx.load(model_path)
    
    # Count operators
    op_counts = Counter()
    for node in model.graph.node:
        op_counts[node.op_type] += 1
    
    print(f"=== {model_path} 内の演算子数 ===")
    for i, (op_type, count) in enumerate(sorted(op_counts.items()), 1):
        print(f"{i:2d}. {op_type}: {count}個")
    
    print(f"\n総演算子数: {sum(op_counts.values())}個")
    print(f"演算子種類数: {len(op_counts)}種類")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python count_operators.py <model.onnx>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    count_operators(model_path)