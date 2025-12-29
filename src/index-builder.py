#!/usr/bin/env python3
"""
快速标准HNSW索引构建器 - 结合批量构建效率和标准HNSW的正确性
生成真正层级稀疏的索引，但构建速度更快
"""

import numpy as np
import faiss
import struct
import time
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set, Tuple
import random
import argparse
import sys
from pathlib import Path

from domain_config import get_config, list_available_configs, SIFTSMALL, LAION, TRIPCLICK, MS_MARCO

class FastStandardHNSWBuilder:
    """快速标准HNSW索引构建器"""
    
    def __init__(self, config=None, seed: int = 42):
        """
        初始化HNSW参数
        
        Args:
            config: DomainConfig对象，包含所有配置参数
            seed: 随机种子
        """
        if config is None:
            raise ValueError("必须提供config参数")
            
        self.config = config
        self.M = config.M
        self.M0 = config.M * 2  # Layer 0的最大连接数
        self.efConstruction = config.efconstruction
        self.ml = 1.0 / np.log(2.0 * config.M)  # 层级分配概率
        self.max_layers = config.layer
        self.seed = seed
        
        # 图结构
        self.graph = {layer: {} for layer in range(self.max_layers + 1)}
        self.node_levels = {}
        self.vectors = None
        self.entry_point = None
        
        # 设置随机种子
        np.random.seed(seed)
        random.seed(seed)
        
        print(f"FastStandardHNSWBuilder初始化:")
        print(f"  数据集: {config.vector_dimension}维, {config.num_docs:,}文档")
        print(f"  M = {self.M} (非0层连接数)")
        print(f"  M0 = {self.M0} (第0层连接数)")
        print(f"  efConstruction = {self.efConstruction}")
        print(f"  efSearch = {config.efsearch}")
        print(f"  ml = {self.ml:.3f} (层级概率)")
        print(f"  最大层数 = {self.max_layers}")
    
    def build_index(self, vectors: np.ndarray):
        """使用FAISS辅助的快速构建方法"""
        self.vectors = vectors
        n_vectors = len(vectors)
        d = vectors.shape[1]
        
        print(f"\n开始快速构建HNSW索引，共{n_vectors}个向量...")
        start_time = time.time()
        
        # 1. 分配层级（符合标准HNSW）
        print("1. 分配节点层级...")
        self._assign_levels(n_vectors)
        
        # 2. 使用FAISS构建辅助索引
        print("2. 构建FAISS辅助索引...")
        faiss_index = faiss.IndexHNSWFlat(d, self.M)
        faiss_index.hnsw.efConstruction = self.efConstruction
        faiss_index.add(vectors)
        
        # 3. 构建层级稀疏的图结构
        print("3. 构建层级稀疏的图结构...")
        self._build_sparse_graph(faiss_index)
        
        build_time = time.time() - start_time
        print(f"\n索引构建完成，耗时: {build_time:.1f}秒")
        
        # 统计信息
        self._print_stats()
    
    def _assign_levels(self, n_vectors: int):
        """分配节点层级，使用标准HNSW的概率分布"""
        level_counts = defaultdict(int)
        
        for i in range(n_vectors):
            level = 0
            while level < self.max_layers and random.random() < self.ml:
                level += 1
            self.node_levels[i] = level
            
            for l in range(level + 1):
                level_counts[l] += 1
        
        # 选择最高层的节点作为入口点
        max_level_nodes = [n for n, l in self.node_levels.items() if l == self.max_layers]
        if max_level_nodes:
            self.entry_point = max_level_nodes[0]
        else:
            # 如果没有最高层节点，选择次高层
            for level in range(self.max_layers - 1, -1, -1):
                level_nodes = [n for n, l in self.node_levels.items() if l == level]
                if level_nodes:
                    self.entry_point = level_nodes[0]
                    break
        
        print(f"  入口点: node_{self.entry_point} (层级: {self.node_levels[self.entry_point]})")
        for level in sorted(level_counts.keys()):
            print(f"  Layer {level}: {level_counts[level]} 节点 ({level_counts[level]/n_vectors*100:.1f}%)")
    
    def _build_sparse_graph(self, faiss_index: faiss.IndexHNSWFlat):
        """构建层级稀疏的图结构"""
        n_vectors = len(self.vectors)
        
        # 设置FAISS搜索参数
        faiss_index.hnsw.efSearch = max(200, self.config.efsearch * 3)
        
        # 分层构建
        for layer in range(self.max_layers + 1):
            print(f"\n  构建Layer {layer}...")
            
            # 获取该层的所有节点
            nodes_in_layer = [i for i in range(n_vectors) if self.node_levels[i] >= layer]
            print(f"    节点数: {len(nodes_in_layer)}")
            
            # 批量搜索邻居
            batch_size = 1000
            for start_idx in range(0, len(nodes_in_layer), batch_size):
                end_idx = min(start_idx + batch_size, len(nodes_in_layer))
                batch_nodes = nodes_in_layer[start_idx:end_idx]
                batch_vectors = self.vectors[batch_nodes]
                
                # 搜索候选邻居
                k = self._get_search_k(layer)
                D, I = faiss_index.search(batch_vectors, k)
                
                # 为每个节点选择邻居
                for i, node_id in enumerate(batch_nodes):
                    # 过滤掉自己和不在该层的节点
                    valid_neighbors = []
                    for j, neighbor in enumerate(I[i]):
                        if neighbor >= 0 and neighbor != node_id and self.node_levels.get(neighbor, -1) >= layer:
                            valid_neighbors.append((D[i][j], neighbor))
                    
                    # 选择邻居
                    selected_neighbors = self._select_neighbors_standard(
                        node_id, valid_neighbors, layer
                    )
                    
                    self.graph[layer][node_id] = selected_neighbors
                
                print(f"\r    进度: {end_idx}/{len(nodes_in_layer)} ({end_idx/len(nodes_in_layer)*100:.1f}%)", end='')
            
            # 统计该层的平均邻居数
            if nodes_in_layer:
                avg_neighbors = np.mean([len(self.graph[layer].get(n, [])) for n in nodes_in_layer])
                print(f"\n    平均邻居数: {avg_neighbors:.1f}")
    
    def _get_search_k(self, layer: int) -> int:
        """根据层级返回搜索的候选数量"""
        base_k = self.config.efconstruction
        if layer == 0:
            return min(base_k * 4, len(self.vectors))  # Layer 0需要更多候选
        elif layer == 1:
            return min(base_k * 2, len(self.vectors))
        else:  # layer >= 2
            return min(base_k, len(self.vectors))
    
    def _select_neighbors_standard(self, node_id: int, candidates: List[Tuple[float, int]], 
                                  layer: int) -> List[int]:
        """
        标准HNSW的邻居选择策略
        高层选择更少、更多样化的邻居
        """
        if not candidates:
            return []
        
        # 根据层级设置目标邻居数（固定3层结构：0,1,2）
        if layer == 0:
            target_neighbors = min(self.M0, len(candidates))  # Layer 0: 最多连接数 (M*2)
        elif layer == 1:
            target_neighbors = min(self.M // 2, len(candidates))  # Layer 1: M/2 连接数
        else:  # layer == 2
            target_neighbors = min(self.M // 4, len(candidates))  # Layer 2: M/4 连接数
        
        # 按距离排序
        candidates.sort()
        
        # 对于高层，实施更严格的多样性选择
        if layer >= 1:
            selected = []
            selected_set = set()
            
            for dist, neighbor in candidates:
                if len(selected) >= target_neighbors:
                    break
                
                # 检查多样性
                should_add = True
                if layer >= 2 and len(selected) > 0:
                    # 高层要求更大的多样性
                    min_diversity_dist = dist * 0.5  # 至少相距50%的距离
                    for s in selected[:5]:  # 检查前几个已选邻居
                        if self._distance(self.vectors[neighbor], self.vectors[s]) < min_diversity_dist:
                            should_add = False
                            break
                
                if should_add and neighbor not in selected_set:
                    selected.append(neighbor)
                    selected_set.add(neighbor)
            
            # 如果选择太少，补充一些最近邻
            if len(selected) < target_neighbors // 2:
                for _, neighbor in candidates:
                    if neighbor not in selected_set:
                        selected.append(neighbor)
                        selected_set.add(neighbor)
                        if len(selected) >= target_neighbors:
                            break
            
            return selected
        else:
            # Layer 0: 简单选择最近邻
            return [neighbor for _, neighbor in candidates[:target_neighbors]]
    
    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算L2距离的平方"""
        return np.sum((a - b) ** 2)
    
    def _print_stats(self):
        """打印索引统计信息"""
        print("\n索引统计信息:")
        
        total_nodes = len(self.vectors)
        for layer in range(self.max_layers + 1):
            nodes_in_layer = [n for n, level in self.node_levels.items() if level >= layer]
            
            if nodes_in_layer:
                neighbor_counts = [len(self.graph[layer].get(n, [])) for n in nodes_in_layer]
                avg_neighbors = np.mean(neighbor_counts)
                min_neighbors = np.min(neighbor_counts)
                max_neighbors = np.max(neighbor_counts)
                
                print(f"  Layer {layer}: {len(nodes_in_layer)} 节点 "
                      f"({len(nodes_in_layer)/total_nodes*100:.1f}%)")
                print(f"    邻居数 - 平均: {avg_neighbors:.1f}, "
                      f"最小: {min_neighbors}, 最大: {max_neighbors}")
            else:
                print(f"  Layer {layer}: 0 节点")
        
        print(f"\n  入口点: node_{self.entry_point} (最高层: {self.node_levels[self.entry_point]})")
    
    def save_trident_format(self, output_dir: str, dataset_name: str = "dataset"):
        """保存为Trident格式"""
        print(f"\n保存Trident格式到: {output_dir}")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        ntotal = len(self.vectors)
        d = self.vectors.shape[1]
        
        # 保存节点文件
        node_file = f"{output_dir}/nodes.bin"
        with open(node_file, 'wb') as f:
            f.write(struct.pack('i', ntotal))
            f.write(struct.pack('i', d))
            f.write(struct.pack('i', self.entry_point))
            
            for i in range(ntotal):
                f.write(struct.pack('i', i))
                f.write(self.vectors[i].tobytes())
        
        print(f"  ✓ 节点文件: {node_file} ({Path(node_file).stat().st_size/1024/1024:.1f} MB)")
        
        # 保存邻居文件
        neighbor_file = f"{output_dir}/neighbors.bin"
        num_levels = self.max_layers + 1
        maxM0 = self.M0  # 使用配置的M0值
        
        with open(neighbor_file, 'wb') as f:
            f.write(struct.pack('i', ntotal))
            f.write(struct.pack('i', num_levels))
            f.write(struct.pack('i', maxM0))
            
            # 为每个节点的每层写入数据
            for node_id in range(ntotal):
                for layer in range(num_levels):
                    f.write(struct.pack('i', node_id))
                    f.write(struct.pack('i', layer))
                    
                    # 获取该节点在该层的邻居
                    neighbors = []
                    if node_id in self.node_levels and self.node_levels[node_id] >= layer:
                        neighbors = self.graph[layer].get(node_id, [])
                    
                    # 填充到maxM0长度，用-1填充
                    padded = list(neighbors) + [-1] * (maxM0 - len(neighbors))
                    
                    for n in padded[:maxM0]:  # 确保不超过maxM0
                        f.write(struct.pack('i', n))
        
        print(f"  ✓ 邻居文件: {neighbor_file} ({Path(neighbor_file).stat().st_size/1024/1024:.1f} MB)")
        
        return node_file, neighbor_file


def read_fvecs(filename):
    """读取fvecs格式文件"""
    fvecs = []
    with open(filename, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec = struct.unpack('f' * dim, f.read(4 * dim))
            fvecs.append(vec)
    return np.array(fvecs).astype('float32')


def main():
    """主函数 - 快速构建标准HNSW索引"""
    parser = argparse.ArgumentParser(description='快速构建标准HNSW索引')
    # 显示可用的数据集配置
    available_configs = list_available_configs()
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=available_configs,
                       help=f'数据集名称，可选: {available_configs}')
    parser.add_argument('--data-path', type=str, help='输入数据文件路径（fvecs格式）')
    parser.add_argument('--output-dir', type=str, help='输出目录（默认: ~/trident/dataset/数据集名/）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（默认: 42）')
    
    args = parser.parse_args()
    
    print("=== 快速标准HNSW索引构建器 ===")
    print("结合批量构建效率和标准HNSW的层级稀疏性\n")
    
    # 获取数据集配置
    config = get_config(args.dataset)
    print(f"使用数据集配置: {args.dataset}")
    print(f"  向量维度: {config.vector_dimension}")
    print(f"  预期文档数: {config.num_docs:,}")
    print(f"  HNSW参数: M={config.M}, efConstruction={config.efconstruction}, layers={config.layer}")
    print()
    
    # 确定数据路径
    if args.data_path:
        base_path = args.data_path
    else:
        # 默认路径
        base_path = f"~/trident/dataset/{args.dataset}/base.fvecs"
    
    # 确定输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"~/trident/dataset/{args.dataset}"
    
    # 加载数据
    print(f"加载数据: {base_path}")
    vectors = read_fvecs(base_path)
    print(f"数据规模: {vectors.shape}")
    
    # 验证数据维度
    if vectors.shape[1] != config.vector_dimension:
        print(f"警告: 数据维度({vectors.shape[1]})与配置维度({config.vector_dimension})不匹配")
    
    # 创建构建器
    builder = FastStandardHNSWBuilder(config=config, seed=args.seed)
    
    # 构建索引
    builder.build_index(vectors)
    
    # 保存为Trident格式
    builder.save_trident_format(output_dir, dataset_name=args.dataset)
    
    print("\n构建完成！")
    print(f"索引文件保存在: {output_dir}/")
    print(f"  - nodes.bin: 节点向量文件")
    print(f"  - neighbors.bin: 邻居关系文件")


if __name__ == "__main__":
    main()