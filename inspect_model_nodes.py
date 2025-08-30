import onnx
import argparse


def inspect_model_nodes(model_path):
    """ONNXモデルのノード構造を詳細に調査"""
    try:
        model = onnx.load(model_path)
        
        print(f"=== {model_path} のノード構造 ===")
        print(f"モデル名: {model.graph.name}")
        print(f"IR Version: {model.ir_version}")
        print(f"Producer: {model.producer_name} {model.producer_version}")
        
        # 入力情報
        print(f"\n=== 入力 ===")
        for input_tensor in model.graph.input:
            shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param for dim in input_tensor.type.tensor_type.shape.dim]
            print(f"名前: {input_tensor.name}")
            print(f"形状: {shape}")
            print(f"タイプ: {input_tensor.type.tensor_type.elem_type}")
        
        # 出力情報
        print(f"\n=== 出力 ===")
        for output_tensor in model.graph.output:
            shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param for dim in output_tensor.type.tensor_type.shape.dim]
            print(f"名前: {output_tensor.name}")
            print(f"形状: {shape}")
            print(f"タイプ: {output_tensor.type.tensor_type.elem_type}")
        
        # 中間テンソル（value_info）
        print(f"\n=== 中間テンソル（Value Info）===")
        print(f"総数: {len(model.graph.value_info)}")
        for i, value_info in enumerate(model.graph.value_info):
            if i < 20:  # 最初の20個のみ表示
                shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param for dim in value_info.type.tensor_type.shape.dim]
                print(f"{i+1:3d}. {value_info.name} - 形状: {shape}")
        if len(model.graph.value_info) > 20:
            print(f"    ... 他 {len(model.graph.value_info) - 20} 個")
        
        # Concatを含むノードを検索
        print(f"\n=== Concatを含むノード ===")
        concat_nodes = []
        for node in model.graph.node:
            if 'Concat' in node.op_type:
                concat_nodes.append(node)
                print(f"演算子: {node.op_type}")
                print(f"名前: {node.name}")
                print(f"入力: {node.input}")
                print(f"出力: {node.output}")
                print("---")
        
        # model.23を含むノードを検索
        print(f"\n=== 'model.23'を含むノード/テンソル ===")
        found_model23 = []
        
        # ノード名を検索
        for node in model.graph.node:
            if 'model.23' in node.name:
                found_model23.append(f"ノード名: {node.name} (演算子: {node.op_type})")
        
        # 入力テンソル名を検索
        for node in model.graph.node:
            for input_name in node.input:
                if 'model.23' in input_name and input_name not in [item.split(': ')[1] for item in found_model23]:
                    found_model23.append(f"入力テンソル: {input_name}")
        
        # 出力テンソル名を検索
        for node in model.graph.node:
            for output_name in node.output:
                if 'model.23' in output_name and output_name not in [item.split(': ')[1] for item in found_model23]:
                    found_model23.append(f"出力テンソル: {output_name}")
        
        # value_infoを検索
        for value_info in model.graph.value_info:
            if 'model.23' in value_info.name:
                found_model23.append(f"中間テンソル: {value_info.name}")
        
        if found_model23:
            for item in found_model23:
                print(f"  {item}")
        else:
            print("  'model.23'を含むノード/テンソルは見つかりませんでした")
        
        # 末尾のConcatノードを探す
        print(f"\n=== 推奨分割ポイント（末尾のConcat系ノード）===")
        potential_split_points = []
        
        for node in model.graph.node:
            if node.op_type in ['Concat', 'Add', 'Mul'] and any('output' in out.lower() for out in node.output):
                for output_name in node.output:
                    potential_split_points.append(output_name)
        
        # 末尾近くのノードの出力も候補として追加
        total_nodes = len(model.graph.node)
        for i, node in enumerate(model.graph.node):
            if i >= total_nodes - 10:  # 最後の10ノード
                for output_name in node.output:
                    if output_name not in potential_split_points:
                        potential_split_points.append(output_name)
        
        if potential_split_points:
            print("候補:")
            for i, point in enumerate(potential_split_points[:10]):  # 最初の10個
                print(f"  {i+1}. {point}")
        else:
            print("  適切な分割ポイントが見つかりませんでした")
            
        return True
        
    except Exception as e:
        print(f"モデル調査エラー: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="ONNXモデルのノード構造を調査")
    parser.add_argument("model_path", help="調査するONNXモデルファイル")
    
    args = parser.parse_args()
    
    inspect_model_nodes(args.model_path)


if __name__ == "__main__":
    main()