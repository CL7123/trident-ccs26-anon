#!/usr/bin/env python3
"""
内存占用分析脚本
计算 Server 和 Client 的内存占用（基于数据文件大小）
"""

import os
import sys
import json
import glob
import logging
from pathlib import Path
from typing import Dict, List

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [MemAnalysis] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class MemoryAnalyzer:
    """内存占用分析器"""

    def __init__(self, base_dir: str = "~/trident/dataset"):
        self.base_dir = base_dir
        self.datasets = ["siftsmall", "nfcorpus", "laion", "tripclick", "ms_marco"]
        self.results = {}

    def get_file_size_mb(self, filepath: str) -> float:
        """获取文件大小（MB）"""
        if os.path.exists(filepath):
            return os.path.getsize(filepath) / (1024 ** 2)
        return 0.0

    def get_file_size_gb(self, filepath: str) -> float:
        """获取文件大小（GB）"""
        if os.path.exists(filepath):
            return os.path.getsize(filepath) / (1024 ** 3)
        return 0.0

    def get_directory_size_gb(self, dirpath: str) -> float:
        """获取目录下所有文件的总大小（GB）"""
        total_size = 0
        if os.path.exists(dirpath):
            for root, dirs, files in os.walk(dirpath):
                for file in files:
                    filepath = os.path.join(root, file)
                    try:
                        total_size += os.path.getsize(filepath)
                    except OSError as e:
                        logger.warning(f"无法访问文件 {filepath}: {e}")
        return total_size / (1024 ** 3)

    def analyze_server_memory(self, dataset: str, server_id: int) -> Dict:
        """分析单个 server 的内存占用"""
        server_data = {
            'nodes_shares_gb': 0.0,
            'neighbor_lists_gb': 0.0,
            'triples_gb': 0.0,
            'total_gb': 0.0,
            'details': []
        }

        dataset_dir = os.path.join(self.base_dir, dataset, f"server_{server_id}")

        # 1. nodes_shares.npy
        nodes_file = os.path.join(dataset_dir, "nodes_shares.npy")
        if os.path.exists(nodes_file):
            size_gb = self.get_file_size_gb(nodes_file)
            server_data['nodes_shares_gb'] = size_gb
            server_data['details'].append(f"nodes_shares.npy: {size_gb:.3f} GB")
            logger.debug(f"  nodes_shares.npy: {size_gb:.3f} GB")
        else:
            logger.warning(f"  nodes_shares.npy 不存在: {nodes_file}")

        # 2. neighbors_shares.npy
        neighbors_file = os.path.join(dataset_dir, "neighbors_shares.npy")
        if os.path.exists(neighbors_file):
            size_gb = self.get_file_size_gb(neighbors_file)
            server_data['neighbor_lists_gb'] = size_gb
            server_data['details'].append(f"neighbors_shares.npy: {size_gb:.3f} GB")
            logger.debug(f"  neighbors_shares.npy: {size_gb:.3f} GB")
        else:
            # 尝试旧格式 neighbor_list_*.npy
            neighbor_pattern = os.path.join(dataset_dir, "neighbor_list_*.npy")
            neighbor_files = glob.glob(neighbor_pattern)
            neighbor_total = 0.0
            for nf in neighbor_files:
                size_gb = self.get_file_size_gb(nf)
                neighbor_total += size_gb
                filename = os.path.basename(nf)
                server_data['details'].append(f"{filename}: {size_gb:.3f} GB")
                logger.debug(f"  {filename}: {size_gb:.3f} GB")
            server_data['neighbor_lists_gb'] = neighbor_total

            if neighbor_total == 0:
                logger.warning(f"  neighbors_shares.npy 不存在: {neighbors_file}")

        # 3. triples (在单独的 triples 目录下)
        triples_dir = os.path.join(self.base_dir, "triples", f"server_{server_id}")
        if os.path.exists(triples_dir):
            triples_pattern = os.path.join(triples_dir, "triples_*.npy")
            triples_files = glob.glob(triples_pattern)
            triples_total = 0.0
            for tf in triples_files:
                size_gb = self.get_file_size_gb(tf)
                triples_total += size_gb
                filename = os.path.basename(tf)
                server_data['details'].append(f"{filename}: {size_gb:.3f} GB")
                logger.debug(f"  {filename}: {size_gb:.3f} GB")

            server_data['triples_gb'] = triples_total
        else:
            logger.warning(f"  triples 目录不存在: {triples_dir}")

        # 计算总计
        server_data['total_gb'] = (server_data['nodes_shares_gb'] +
                                   server_data['neighbor_lists_gb'] +
                                   server_data['triples_gb'])

        return server_data

    def analyze_client_memory(self, dataset: str) -> Dict:
        """分析 client 的内存占用"""
        client_data = {
            'nodes_mb': 0.0,
            'dpf_keys_mb': 0.0,
            'total_mb': 0.0,
            'details': []
        }

        dataset_dir = os.path.join(self.base_dir, dataset)

        # 1. 原始节点向量（用于验证）
        # 尝试 nodes.bin 或 nodes.npy
        nodes_file_bin = os.path.join(dataset_dir, "nodes.bin")
        nodes_file_npy = os.path.join(dataset_dir, "nodes.npy")

        if os.path.exists(nodes_file_bin):
            size_mb = self.get_file_size_mb(nodes_file_bin)
            client_data['nodes_mb'] = size_mb
            client_data['details'].append(f"nodes.bin: {size_mb:.2f} MB")
            logger.debug(f"  nodes.bin: {size_mb:.2f} MB")
        elif os.path.exists(nodes_file_npy):
            size_mb = self.get_file_size_mb(nodes_file_npy)
            client_data['nodes_mb'] = size_mb
            client_data['details'].append(f"nodes.npy: {size_mb:.2f} MB")
            logger.debug(f"  nodes.npy: {size_mb:.2f} MB")
        else:
            logger.warning(f"  nodes file 不存在: {nodes_file_bin} 或 {nodes_file_npy}")

        # 2. DPF keys 大小（从之前的测试结果获取）
        # 尝试读取 client_cost 测试结果
        cost_files = glob.glob(f"client_cost_{dataset}_*.json")
        if cost_files:
            latest_file = sorted(cost_files)[-1]
            try:
                with open(latest_file, 'r') as f:
                    data = json.load(f)
                    if 'statistics' in data and 'key_size' in data['statistics']:
                        # key_size 是单个key的大小，需要 × 3 (三个servers)
                        single_key_kb = data['statistics']['key_size']['mean']
                        dpf_keys_mb = (single_key_kb * 3) / 1024  # KB → MB
                        client_data['dpf_keys_mb'] = dpf_keys_mb
                        client_data['details'].append(f"DPF keys (3×): {dpf_keys_mb:.2f} MB")
                        logger.debug(f"  DPF keys: {dpf_keys_mb:.2f} MB (from {latest_file})")
            except Exception as e:
                logger.warning(f"  无法读取 DPF key 大小: {e}")
        else:
            logger.warning(f"  未找到 {dataset} 的 client_cost 测试结果，DPF key 大小设为 0")

        # 计算总计
        client_data['total_mb'] = client_data['nodes_mb'] + client_data['dpf_keys_mb']

        return client_data

    def analyze_dataset(self, dataset: str) -> Dict:
        """分析单个数据集的内存占用"""
        logger.info(f"\n{'='*80}")
        logger.info(f"分析数据集: {dataset.upper()}")
        logger.info(f"{'='*80}")

        result = {
            'dataset': dataset,
            'servers': {},
            'server_avg_gb': 0.0,
            'client': {},
            'exists': False
        }

        dataset_dir = os.path.join(self.base_dir, dataset)
        if not os.path.exists(dataset_dir):
            logger.warning(f"数据集目录不存在: {dataset_dir}")
            return result

        result['exists'] = True

        # 分析3个servers
        logger.info("\nServer 端内存占用:")
        server_totals = []
        for server_id in [1, 2, 3]:
            logger.info(f"\n  Server {server_id}:")
            server_data = self.analyze_server_memory(dataset, server_id)
            result['servers'][server_id] = server_data
            server_totals.append(server_data['total_gb'])

            logger.info(f"    节点分享: {server_data['nodes_shares_gb']:.3f} GB")
            logger.info(f"    邻居列表: {server_data['neighbor_lists_gb']:.3f} GB")
            logger.info(f"    三元组: {server_data['triples_gb']:.3f} GB")
            logger.info(f"    总计: {server_data['total_gb']:.3f} GB")

        # 计算平均值
        if server_totals:
            result['server_avg_gb'] = sum(server_totals) / len(server_totals)
            logger.info(f"\n  Server 平均内存: {result['server_avg_gb']:.3f} GB")

        # 分析 Client
        logger.info(f"\nClient 端内存占用:")
        client_data = self.analyze_client_memory(dataset)
        result['client'] = client_data

        logger.info(f"  节点向量: {client_data['nodes_mb']:.2f} MB")
        logger.info(f"  DPF keys: {client_data['dpf_keys_mb']:.2f} MB")
        logger.info(f"  总计: {client_data['total_mb']:.2f} MB")

        return result

    def run_analysis(self):
        """运行完整的内存分析"""
        logger.info("="*80)
        logger.info("开始内存占用分析")
        logger.info("="*80)

        for dataset in self.datasets:
            result = self.analyze_dataset(dataset)
            if result['exists']:
                self.results[dataset] = result

        self.print_summary()
        self.save_results()

    def print_summary(self):
        """打印汇总表格"""
        logger.info(f"\n\n{'='*100}")
        logger.info("内存占用汇总")
        logger.info(f"{'='*100}\n")

        # 表格标题
        logger.info(f"{'Dataset':<15} {'Server (GB)':<15} {'Client (MB)':<15} {'Compass Server (GB)':<20} {'Compass Client (MB)':<20}")
        logger.info(f"{'-'*100}")

        # Compass 数据（用于对比）
        compass_data = {
            'laion': {'server': 0.95, 'client': 5.49},
            'siftsmall': {'server': 6.19, 'client': 35.84},
            'tripclick': {'server': 24.19, 'client': 77.48},
            'ms_marco': {'server': 193.50, 'client': 498.65}
        }

        for dataset, result in self.results.items():
            server_gb = result['server_avg_gb']
            client_mb = result['client']['total_mb']

            # 获取 Compass 数据
            compass = compass_data.get(dataset, {'server': '-', 'client': '-'})
            compass_server = compass['server'] if isinstance(compass['server'], str) else f"{compass['server']:.2f}"
            compass_client = compass['client'] if isinstance(compass['client'], str) else f"{compass['client']:.2f}"

            logger.info(f"{dataset.upper():<15} {server_gb:<15.3f} {client_mb:<15.2f} {compass_server:<20} {compass_client:<20}")

        logger.info(f"{'='*100}\n")

        # Markdown 格式
        logger.info("\nMarkdown 表格格式:")
        logger.info("```")
        logger.info("| Dataset   | Trident Server (GB) | Trident Client (MB) | Compass Server (GB) | Compass Client (MB) |")
        logger.info("|-----------|---------------------|---------------------|---------------------|---------------------|")

        for dataset, result in self.results.items():
            server_gb = result['server_avg_gb']
            client_mb = result['client']['total_mb']
            compass = compass_data.get(dataset, {'server': '-', 'client': '-'})

            logger.info(f"| {dataset.upper():<9} | {server_gb:<19.2f} | {client_mb:<19.2f} | "
                       f"{compass['server'] if isinstance(compass['server'], str) else f'{compass['server']:.2f}':<19} | "
                       f"{compass['client'] if isinstance(compass['client'], str) else f'{compass['client']:.2f}':<19} |")

        logger.info("```\n")

    def save_results(self):
        """保存结果到 JSON 文件"""
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"memory_analysis_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"详细结果已保存到: {filename}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='内存占用分析')
    parser.add_argument('--base-dir', type=str, default='~/trident/dataset',
                       help='数据集基础目录')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细日志')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    analyzer = MemoryAnalyzer(base_dir=args.base_dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
