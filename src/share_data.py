#!/usr/bin/env python3
"""
将LAION数据集进行秘密共享
- 节点向量：100k × 512维
- 邻居列表：线性化存储 (node_id * 3 + layer)
"""

import sys
import os
import struct
import numpy as np
import json
import time
import argparse
from typing import Dict, Tuple

# 导入域配置
from domain_config import get_config as get_domain_config, list_available_configs, SIFTSMALL, LAION, TRIPCLICK, MS_MARCO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add parent directory to path for imports
from basic_functionalities import MPC23SSS, get_config as get_mpc_config


class DatasetLoader:
    """加载和解析数据集"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        
    def load_nodes(self) -> np.ndarray:
        """加载节点向量数据"""
        print("加载节点向量...")
        nodes_path = os.path.join(self.data_dir, "nodes.bin")
        
        with open(nodes_path, 'rb') as f:
            # 读取header
            num_nodes = struct.unpack('<I', f.read(4))[0]
            vector_dim = struct.unpack('<I', f.read(4))[0]
            
            print(f"  节点数: {num_nodes:,}")
            print(f"  向量维度: {vector_dim}")
            
            # 跳过额外的header字节
            f.read(4)  # 通常是12或其他元数据
            
            # 读取所有向量
            total_floats = num_nodes * vector_dim
            vectors = struct.unpack(f'<{total_floats}f', f.read(total_floats * 4))
            
            # 重塑为二维数组
            nodes = np.array(vectors, dtype=np.float32).reshape(num_nodes, vector_dim)
            
        return nodes
    
    def load_neighbors(self) -> Tuple[np.ndarray, int, int]:
        """加载邻居列表数据"""
        print("\n加载邻居列表...")
        neighbors_path = os.path.join(self.data_dir, "neighbors.bin")
        
        with open(neighbors_path, 'rb') as f:
            # 读取header
            num_nodes = struct.unpack('<I', f.read(4))[0]
            num_layers = struct.unpack('<I', f.read(4))[0]
            max_neighbors = struct.unpack('<I', f.read(4))[0]
            _ = struct.unpack('<I', f.read(4))[0]  # 跳过额外的0
            
            print(f"  节点数: {num_nodes:,}")
            print(f"  层数: {num_layers}")
            print(f"  最大邻居数: {max_neighbors}")
            
            # 数据格式：每个节点有 (3层 × 128邻居 + 2个额外值) 个整数
            ints_per_node = num_layers * max_neighbors + 2
            
            # 读取所有数据
            total_ints = num_nodes * ints_per_node
            print(f"  读取 {total_ints:,} 个整数...")
            all_data = struct.unpack(f'<{total_ints}I', f.read(total_ints * 4))
            
            # 解析为结构化数据
            neighbors = []
            
            for node_id in range(num_nodes):
                # 计算该节点数据的起始位置
                start_idx = node_id * ints_per_node
                
                node_neighbors = []
                for layer in range(num_layers):
                    # 获取该层的128个邻居
                    layer_start = start_idx + layer * max_neighbors
                    layer_data = all_data[layer_start:layer_start + max_neighbors]
                    
                    # 过滤掉填充值（-1表示为4294967295）
                    actual_neighbors = [x for x in layer_data if x != 4294967295 and x < num_nodes]
                    node_neighbors.append(actual_neighbors)
                
                neighbors.append(node_neighbors)
                
                # 进度显示
                if (node_id + 1) % 10000 == 0:
                    print(f"  已处理 {node_id + 1:,} / {num_nodes:,} 个节点")
            
            return neighbors, num_layers, max_neighbors


class DatasetSecretSharing:
    """对数据集进行秘密共享"""
    
    def __init__(self, dataset_name: str):
        # 获取域配置
        self.domain_config = get_domain_config(dataset_name)
        
        # 使用域配置创建MPC配置
        self.mpc_config = get_mpc_config(dataset_name)
        self.mpc = MPC23SSS(self.mpc_config)
        
        # 使用域配置中的参数
        self.field_size = self.domain_config.prime
        # 根据输出位数动态设置缩放因子
        # 对于不同数据集使用不同的缩放策略
        if dataset_name == "siftsmall":
            # siftsmall数据范围[0, 180]，使用较小的缩放因子
            self.scale_factor = 2 ** 20  # 足够存储180的精度
        elif dataset_name == "tripclick":
            # tripclick数据范围约[-5, 5]，使用较小的缩放因子避免溢出
            self.scale_factor = 2 ** 20  # 1M的缩放因子，减小以避免768维向量计算时溢出
        elif dataset_name == "nfcorpus":
            # nfcorpus数据范围约[-12, 4]，使用较小的缩放因子
            self.scale_factor = 2 ** 22  # 4M的缩放因子，避免溢出
        else:
            # 其他数据集（如LAION）数据范围[-1, 1]，使用原始缩放因子
            self.scale_factor = 2 ** (self.domain_config.output_bits - 2)
        
    def float_to_field(self, value: float) -> int:
        """将浮点数转换到有限域"""
        # 缩放并转为整数，然后对field_size取模
        # Python的模运算会自动处理负数，将其映射到[0, field_size)范围
        scaled = int(value * self.scale_factor)
        return scaled % self.field_size
    
    def share_nodes(self, nodes: np.ndarray) -> Dict[int, np.ndarray]:
        """对节点向量进行秘密共享"""
        num_nodes, vector_dim = nodes.shape
        print(f"\n开始节点向量秘密共享...")
        print(f"  数据形状: {nodes.shape}")
        
        # 初始化服务器份额数组
        server_shares = {
            1: np.zeros((num_nodes, vector_dim), dtype=np.uint32),
            2: np.zeros((num_nodes, vector_dim), dtype=np.uint32),
            3: np.zeros((num_nodes, vector_dim), dtype=np.uint32)
        }
        
        # 进度显示
        start_time = time.time()
        
        for i in range(num_nodes):
            # 对每个向量的每个维度进行秘密共享
            for j in range(vector_dim):
                # 转换到有限域
                field_value = self.float_to_field(nodes[i, j])
                
                # 生成秘密份额
                shares = self.mpc.share_secret(field_value)
                
                # 分配给各服务器
                for server_id in range(1, 4):
                    server_shares[server_id][i, j] = shares[server_id-1].value
            
            # 进度显示
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (num_nodes - i - 1) / rate
                print(f"  进度: {i+1:,}/{num_nodes:,} ({(i+1)/num_nodes*100:.1f}%) - "
                      f"速率: {rate:.0f} 节点/秒 - 剩余: {eta:.0f}秒")
        
        return server_shares
    
    def share_neighbors_linearized(self, neighbors: list, num_layers: int, 
                                   max_neighbors: int) -> Dict[int, np.ndarray]:
        """对邻居列表进行线性化和秘密共享"""
        num_nodes = len(neighbors)
        total_entries = num_nodes * num_layers
        
        print(f"\n开始邻居列表秘密共享...")
        print(f"  线性化数组大小: {total_entries:,} × {max_neighbors}")
        
        # 初始化服务器份额数组
        server_shares = {
            1: np.zeros((total_entries, max_neighbors), dtype=np.uint32),
            2: np.zeros((total_entries, max_neighbors), dtype=np.uint32),
            3: np.zeros((total_entries, max_neighbors), dtype=np.uint32)
        }
        
        # 进度显示
        start_time = time.time()
        processed = 0
        
        for node_id in range(num_nodes):
            for layer in range(num_layers):
                # 计算线性索引
                linear_idx = node_id * num_layers + layer
                
                # 获取该位置的邻居列表
                neighbor_list = neighbors[node_id][layer] if layer < len(neighbors[node_id]) else []
                
                # Padding到max_neighbors，使用-1作为填充值
                padding_value = self.field_size - 1  # -1 在有限域中的表示
                padded_neighbors = neighbor_list + [padding_value] * (max_neighbors - len(neighbor_list))
                
                # 对每个邻居ID进行秘密共享
                for j in range(max_neighbors):
                    neighbor_id = padded_neighbors[j]
                    
                    # 生成秘密份额
                    shares = self.mpc.share_secret(neighbor_id)
                    
                    # 分配给各服务器
                    for server_id in range(1, 4):
                        server_shares[server_id][linear_idx, j] = shares[server_id-1].value
                
                processed += 1
                
                # 进度显示
                if processed % 10000 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed
                    eta = (total_entries - processed) / rate
                    print(f"  进度: {processed:,}/{total_entries:,} ({processed/total_entries*100:.1f}%) - "
                          f"速率: {rate:.0f} 条目/秒 - 剩余: {eta:.0f}秒")
        
        return server_shares


def main():
    """主函数：加载数据并进行秘密共享"""
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='对数据集进行秘密共享')
    available_configs = list_available_configs()
    parser.add_argument('--dataset', type=str, default=SIFTSMALL,
                        choices=available_configs,
                        help=f'数据集名称，可选: {available_configs}')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='数据集目录路径')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录路径')
    args = parser.parse_args()
    
    print(f"=== 数据集秘密共享: {args.dataset} ===\n")
    
    # 获取域配置
    domain_config = get_domain_config(args.dataset)
    print(f"使用数据集配置: {args.dataset}")
    print(f"  向量维度: {domain_config.vector_dimension}")
    print(f"  预期文档数: {domain_config.num_docs:,}")
    print(f"  素数域: p = {domain_config.prime:,}")
    print(f"  输出位数: {domain_config.output_bits}")
    print()
    
    # 设置默认路径
    # Use relative path from project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.data_dir is None:
        data_dir = os.path.join(project_root, "datasets", args.dataset)
    else:
        data_dir = args.data_dir

    if args.output_dir is None:
        output_dir = os.path.join(project_root, "datasets", args.dataset)
    else:
        output_dir = args.output_dir
    
    # 1. 加载数据
    loader = DatasetLoader(data_dir)
    
    # 加载节点向量
    nodes = loader.load_nodes()
    
    # 加载邻居列表
    neighbors, num_layers, max_neighbors = loader.load_neighbors()
    
    # 2. 进行秘密共享
    sharer = DatasetSecretSharing(args.dataset)
    
    # 共享节点向量
    node_shares = sharer.share_nodes(nodes)
    
    # 共享邻居列表
    neighbor_shares = sharer.share_neighbors_linearized(neighbors, num_layers, max_neighbors)
    
    # 3. 保存到文件
    
    print("\n保存秘密份额...")
    for server_id in range(1, 4):
        server_dir = os.path.join(output_dir, f"server_{server_id}")
        os.makedirs(server_dir, exist_ok=True)
        
        # 保存节点份额
        nodes_path = os.path.join(server_dir, "nodes_shares.npy")
        np.save(nodes_path, node_shares[server_id])
        print(f"  Server {server_id} 节点份额: {nodes_path}")
        
        # 保存邻居份额
        neighbors_path = os.path.join(server_dir, "neighbors_shares.npy")
        np.save(neighbors_path, neighbor_shares[server_id])
        print(f"  Server {server_id} 邻居份额: {neighbors_path}")
        
        # 保存元数据
        metadata = {
            "dataset": args.dataset,
            "num_nodes": len(nodes),
            "num_layers": num_layers,
            "max_neighbors": max_neighbors,
            "vector_dim": nodes.shape[1],
            "indexing": "linear",
            "index_formula": "node_id * num_layers + layer",
            "scale_factor": sharer.scale_factor,
            "field_size": sharer.field_size,
            "domain_bits": domain_config.domain_bits,
            "domain_size": domain_config.domain_size,
            "output_bits": domain_config.output_bits,
            "M": domain_config.M,
            "efconstruction": domain_config.efconstruction,
            "efsearch": domain_config.efsearch,
            "total_neighbor_entries": len(nodes) * num_layers,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        metadata_path = os.path.join(server_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Server {server_id} 元数据: {metadata_path}")
    
    print(f"\n✅ {args.dataset}数据集秘密共享完成！")
    print(f"\n输出目录: {output_dir}")
    print("使用示例:")
    print("  python share_data.py --dataset siftsmall")
    print("  python share_data.py --dataset laion")
    print("  python share_data.py --dataset tripclick --output-dir /custom/path")


if __name__ == "__main__":
    main()