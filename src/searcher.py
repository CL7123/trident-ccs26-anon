
#!/usr/bin/env python3
"""
测试和修复TridentSearcher的搜索算法
"""

import numpy as np
import heapq
import time
import struct
import argparse
import os
from typing import List, Tuple
from datetime import datetime

from domain_config import get_config, list_available_configs, SIFTSMALL, LAION, TRIPCLICK, MS_MARCO

class TridentSearcher:
    """Trident搜索器"""
    
    def __init__(self, node_file: str, neighbor_file: str, config=None):
        """初始化搜索器"""
        self.config = config
        self.vectors = None
        self.graph = {}
        self.node_levels = {}
        self.entry_point = 0
        
        self._load_nodes(node_file) 
        self._load_neighbors(neighbor_file)
        
        print(f"TridentSearcher初始化完成:")
        print(f"  向量数: {len(self.vectors)}")
        print(f"  维度: {self.vectors.shape[1]}")
        print(f"  层数: {len(self.graph)}")
        print(f"  入口点: {self.entry_point}")
        if self.config:
            print(f"  配置 efSearch: {self.config.efsearch}")
            print(f"  预期文档数: {self.config.num_docs:,}")
    
    def _load_nodes(self, filename: str):
        """加载节点向量"""
        with open(filename, 'rb') as f:
            ntotal = struct.unpack('i', f.read(4))[0]
            d = struct.unpack('i', f.read(4))[0]
            self.entry_point = struct.unpack('i', f.read(4))[0]
            
            vectors = []
            for i in range(ntotal):
                node_id = struct.unpack('i', f.read(4))[0]
                vector = np.frombuffer(f.read(d * 4), dtype=np.float32)
                vectors.append(vector)
            
            self.vectors = np.array(vectors)
    
    def _load_neighbors(self, filename: str):
        """加载邻居关系"""
        with open(filename, 'rb') as f:
            ntotal = struct.unpack('i', f.read(4))[0]
            num_levels = struct.unpack('i', f.read(4))[0]
            maxM0 = struct.unpack('i', f.read(4))[0]
            
            for layer in range(num_levels):
                self.graph[layer] = {}
            
            for i in range(ntotal):
                for layer in range(num_levels):
                    node_id = struct.unpack('i', f.read(4))[0]
                    layer_id = struct.unpack('i', f.read(4))[0]
                    
                    neighbors = []
                    for _ in range(maxM0):
                        n = struct.unpack('i', f.read(4))[0]
                        if n >= 0:
                            neighbors.append(n)
                    
                    if neighbors:
                        self.graph[layer][node_id] = neighbors
                        
                        if node_id not in self.node_levels:
                            self.node_levels[node_id] = layer
                        else:
                            self.node_levels[node_id] = max(self.node_levels[node_id], layer)
    
    def search_single(self, query: np.ndarray, k: int, ef: int = None, return_stats: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """单个查询搜索

        Args:
            query: 查询向量
            k: 返回的最近邻数量
            ef: efSearch参数
            return_stats: 是否返回统计信息

        Returns:
            distances, indices 或 (distances, indices, stats)
        """
        # 使用配置的efSearch作为默认值
        if ef is None:
            ef = self.config.efsearch if self.config else 32

        # 初始化统计信息
        self.nodes_visited = set()
        self.neighborlists_accessed = 0

        # 1. 从入口点开始，在高层进行贪心搜索
        curr_nearest = self.entry_point
        curr_level = self._get_node_level(self.entry_point)

        # 从最高层向下搜索到Layer 1
        for level in range(curr_level, 0, -1):
            nearest = self._greedy_search_layer(query, curr_nearest, 1, level)
            if nearest:
                curr_nearest = nearest[0]

        # 2. 在Layer 0进行扩展搜索
        candidates = self._search_layer(query, [curr_nearest], ef, 0)

        # 3. 提取top-k结果
        distances, indices = self._get_top_k(candidates, k)

        if return_stats:
            stats = {
                'nodes_visited': len(self.nodes_visited),
                'neighborlists_accessed': self.neighborlists_accessed
            }
            return distances, indices, stats

        return distances, indices
    
    # def _greedy_search_layer(self, query: np.ndarray, entry_point: int, 
    #                         num_closest: int, layer: int) -> List[int]:
    #     """在指定层进行贪心搜索，只返回最近的num_closest个节点"""
    #     visited = set()
    #     candidates = []  # (distance, node_id)
    #     W = []  # 结果集
    #     
    #     # 初始化
    #     visited.add(entry_point)
    #     d = self._distance(query, self.vectors[entry_point])
    #     heapq.heappush(candidates, (d, entry_point))
    #     W.append((d, entry_point))
    #     
    #     while candidates:
    #         curr_dist, curr = heapq.heappop(candidates)
    #         
    #         # 如果当前节点比结果集中最近的还远，停止搜索
    #         if curr_dist > W[0][0]:
    #             break
    #         
    #         # 检查邻居
    #         neighbors = self._get_neighbors(curr, layer)
    #         for neighbor in neighbors:
    #             if neighbor not in visited:
    #                 visited.add(neighbor)
    #                 d = self._distance(query, self.vectors[neighbor])
    #                 
    #                 if d < W[0][0] or len(W) < num_closest:
    #                     heapq.heappush(candidates, (d, neighbor))
    #                     W.append((d, neighbor))
    #                     W.sort(key=lambda x: x[0])
    #                     if len(W) > num_closest:
    #                         W.pop()
    #     
    #     return [node for dist, node in W[:num_closest]]
    
    def _greedy_search_layer(self, query: np.ndarray, entry_point: int,
                            num_closest: int, layer: int) -> List[int]:
        """真正的贪心搜索 - 在高层快速导航"""
        current = entry_point
        current_dist = self._distance(query, self.vectors[current])

        # 记录访问的节点
        self.nodes_visited.add(current)

        # 记录访问路径（用于调试）
        path = [current]

        # 真正的贪心：不断寻找更近的邻居
        improved = True
        while improved:
            improved = False

            # 获取当前节点的邻居
            neighbors = self._get_neighbors(current, layer)
            self.neighborlists_accessed += 1  # 统计neighborlist访问

            # 贪心选择：找到第一个更近的邻居就立即移动
            for neighbor in neighbors:
                self.nodes_visited.add(neighbor)  # 记录访问的邻居
                neighbor_dist = self._distance(query, self.vectors[neighbor])

                if neighbor_dist < current_dist:
                    # 找到更近的节点，立即移动
                    current = neighbor
                    current_dist = neighbor_dist
                    path.append(current)
                    improved = True
                    break  # 关键：立即停止检查其他邻居

        # 返回最终到达的节点
        return [current]
    
    def _search_layer(self, query: np.ndarray, entry_points: List[int],
                              ef: int, layer: int) -> List[Tuple[int, float]]:
        """层搜索算法"""
        visited = set()
        candidates = []  # 最小堆
        W = []  # 结果集，保持有序

        # 初始化
        for point in entry_points:
            if point not in visited:
                visited.add(point)
                self.nodes_visited.add(point)  # 统计访问的节点
                d = self._distance(query, self.vectors[point])
                heapq.heappush(candidates, (d, point))
                W.append((d, point))

        # 主循环
        while candidates:
            curr_dist, curr = heapq.heappop(candidates)

            # 动态更新的早停条件
            if W:
                W.sort(key=lambda x: x[0])
                if len(W) >= ef and curr_dist > W[ef-1][0]:
                    break

            # 探索邻居
            neighbors = self._get_neighbors(curr, layer)
            self.neighborlists_accessed += 1  # 统计neighborlist访问

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    self.nodes_visited.add(neighbor)  # 统计访问的节点
                    d = self._distance(query, self.vectors[neighbor])

                    # 更灵活的加入条件
                    if len(W) < ef or d < W[-1][0]:
                        heapq.heappush(candidates, (d, neighbor))
                        W.append((d, neighbor))

                        # 保持W有序并限制大小
                        if len(W) > ef:
                            W.sort(key=lambda x: x[0])
                            W = W[:ef]

        # 返回最终结果
        W.sort(key=lambda x: x[0])
        return [(node, dist) for dist, node in W]
    
    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算L2距离的平方"""
        return np.sum((a - b) ** 2)
    
    def _get_neighbors(self, node: int, layer: int) -> List[int]:
        """获取节点在指定层的邻居"""
        if layer in self.graph and node in self.graph[layer]:
            return self.graph[layer][node]
        return []
    
    def _get_node_level(self, node: int) -> int:
        """获取节点的最高层级"""
        return self.node_levels.get(node, 0)
    
    def _get_top_k(self, candidates: List[Tuple[int, float]], k: int) -> Tuple[np.ndarray, np.ndarray]:
        """从候选集中提取top-k结果"""
        distances = np.full(k, float('inf'), dtype=np.float32)
        indices = np.full(k, -1, dtype=np.int32)
        
        for i in range(min(k, len(candidates))):
            indices[i] = candidates[i][0]
            distances[i] = candidates[i][1]
        
        return distances, indices

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

def read_ivecs(filename):
    """读取ivecs格式的ground truth"""
    ivecs = []
    with open(filename, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec = struct.unpack('i' * dim, f.read(4 * dim))
            ivecs.append(vec)
    return np.array(ivecs)

def calculate_mrr_at_k(results, ground_truth, k=10):
    """计算MRR@k"""
    mrr_sum = 0.0
    num_queries = min(len(results), len(ground_truth))
    
    for i in range(num_queries):
        result = results[i][:k]
        gt = ground_truth[i]
        
        for rank, retrieved_id in enumerate(result):
            if retrieved_id in gt[:k]:
                mrr_sum += 1.0 / (rank + 1)
                break
    
    return mrr_sum / num_queries if num_queries > 0 else 0.0

def main():
    """测试改进的搜索算法"""
    # 准备结果文件，使用时间戳避免覆盖
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f"~/trident/result_{timestamp}.md"
    results_content = []
    
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='测试HNSW搜索算法')
    available_configs = list_available_configs()
    parser.add_argument('--dataset', type=str, default=SIFTSMALL,
                        choices=available_configs,
                        help=f'数据集名称，可选: {available_configs}')
    parser.add_argument('--num-queries', type=int, default=100,
                        help='测试的查询数量 (默认: 100)')
    parser.add_argument('--k', type=int, default=10,
                        help='返回的最近邻数量 (默认: 10)')
    args = parser.parse_args()
    
    print(f"=== 测试改进的搜索算法 - 数据集: {args.dataset} ===\n")
    
    # 开始记录结果
    results_content.append(f"# Trident Search Results\n")
    results_content.append(f"**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    results_content.append(f"**数据集**: {args.dataset}\n")
    
    # 获取数据集配置
    config = get_config(args.dataset)
    print(f"使用数据集配置: {args.dataset}")
    print(f"  向量维度: {config.vector_dimension}")
    print(f"  预期文档数: {config.num_docs:,}")
    print(f"  efSearch: {config.efsearch}")
    print()
    
    # 记录配置信息
    results_content.append(f"\n## 配置信息\n")
    results_content.append(f"- 向量维度: {config.vector_dimension}\n")
    results_content.append(f"- 文档数: {config.num_docs:,}\n")
    results_content.append(f"- HNSW参数: M={config.M}, efConstruction={config.efconstruction}\n")
    results_content.append(f"- 默认efSearch: {config.efsearch}\n")
    results_content.append(f"- 查询数: {args.num_queries}\n")
    results_content.append(f"- k: {args.k}\n")
    
    # 构建数据路径（与 index-builder.py 输出一致）
    base_path = f"~/trident/dataset/{args.dataset}"
    
    # 检查路径是否存在
    if not os.path.exists(base_path):
        print(f"错误: 数据集路径不存在: {base_path}")
        print(f"可用的数据集目录:")
        dataset_dir = "~/trident/dataset"
        if os.path.exists(dataset_dir):
            for d in os.listdir(dataset_dir):
                if os.path.isdir(os.path.join(dataset_dir, d)):
                    print(f"  - {d}")
        return
    
    # 加载数据
    query_file = f"{base_path}/query.fvecs"
    gt_file = f"{base_path}/gt.ivecs"
    node_file = f"{base_path}/nodes.bin"
    neighbor_file = f"{base_path}/neighbors.bin"
    
    # 检查必要文件是否存在
    required_files = [query_file, gt_file, node_file, neighbor_file]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"错误: 缺少必要的文件:")
        for f in missing_files:
            print(f"  - {f}")
        return
    
    queries = read_fvecs(query_file)
    ground_truth = read_ivecs(gt_file)
    
    print(f"查询数据: {queries.shape}")
    print(f"Ground truth: {ground_truth.shape}")
    print(f"节点文件: {node_file}")
    print(f"邻居文件: {neighbor_file}")
    print(f"\n测试当前版本...")
    
    # 记录数据信息
    results_content.append(f"\n## 数据信息\n")
    results_content.append(f"- 查询向量: {queries.shape}\n")
    results_content.append(f"- Ground truth: {ground_truth.shape}\n")
        
    # 创建搜索器
    searcher = TridentSearcher(node_file, neighbor_file, config=config)
        
    # 记录测试结果
    results_content.append(f"\n## 测试结果\n")
    results_content.append(f"| efSearch | MRR@{args.k} | 平均延迟 (ms) | 平均访问节点数 | 平均访问邻居列表数 |\n")
    results_content.append(f"|----------|-----------|---------------|----------------|--------------------|\n")
    
    # 测试不同的ef值（基于配置动态设置）
    base_ef = config.efsearch
    test_efs = [base_ef // 2, base_ef, base_ef * 2, base_ef * 4]
    test_efs = [ef for ef in test_efs if ef > 0]  # 过滤掉非正数
    
    for ef in test_efs:
        print(f"\n  efSearch = {ef}:")

        # 预热
        for _ in range(5):
            _ = searcher.search_single(queries[0], args.k, ef)

        # 测试
        all_results = []
        all_stats = []
        start_time = time.time()

        num_queries = min(args.num_queries, len(queries))
        for i in range(num_queries):
            _, indices, stats = searcher.search_single(queries[i], args.k, ef, return_stats=True)
            all_results.append(indices)
            all_stats.append(stats)

        elapsed = time.time() - start_time

        # 计算MRR
        mrr = calculate_mrr_at_k(all_results, ground_truth[:num_queries], k=args.k)
        avg_latency = elapsed / num_queries * 1000

        # 计算平均访问统计
        avg_nodes = np.mean([s['nodes_visited'] for s in all_stats])
        avg_neighborlists = np.mean([s['neighborlists_accessed'] for s in all_stats])

        print(f"    MRR@{args.k}: {mrr:.4f}")
        print(f"    平均延迟: {avg_latency:.2f}ms")
        print(f"    平均访问节点数: {avg_nodes:.1f}")
        print(f"    平均访问邻居列表数: {avg_neighborlists:.1f}")

        # 记录结果
        results_content.append(f"| {ef} | {mrr:.4f} | {avg_latency:.2f} | {avg_nodes:.1f} | {avg_neighborlists:.1f} |\n")

    # 添加最佳结果总结
    results_content.append(f"\n## 总结\n")
    results_content.append(f"- 测试完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    results_content.append(f"- 数据集: {args.dataset}\n")
    results_content.append(f"- 索引位置: {base_path}\n")
    
    # 打印结果内容，让用户手动保存
    print(f"\n{'='*60}")
    print("测试结果（可手动复制保存）:")
    print('='*60)
    for line in results_content:
        print(line.rstrip())
    print('='*60)

if __name__ == "__main__":
    main()