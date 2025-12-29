#!/usr/bin/env python3
"""
客户端成本分析测试脚本
测量 DPF key 生成、重构、距离计算等客户端侧的开销
"""

import sys
import os
import time
import json
import random
import logging
import argparse
import numpy as np
import psutil
from typing import Dict, List
from collections import defaultdict

# 添加路径
sys.path.append('~/trident/distributed-deploy')
sys.path.append('~/trident/src')

from client import DistributedClient
from config import SERVERS  # 导入当前目录的配置（新IP）

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [ClientCost] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ClientCostBenchmark:
    """客户端成本测试"""

    def __init__(self, dataset: str = "siftsmall"):
        self.dataset = dataset
        self.client = DistributedClient(dataset=dataset, servers_config=SERVERS)
        self.process = psutil.Process()
        self.results = []

        # 连接到服务器
        if not self.client.connect_to_servers():
            raise RuntimeError("无法连接到服务器")

        logger.info(f"客户端成本测试初始化完成 - 数据集: {dataset}")

    def measure_single_query(self, node_id: int) -> Dict:
        """测量单个查询的各项成本指标"""
        metrics = {
            'node_id': node_id,
            'key_gen_time_ms': 0,
            'key_size_kb': 0,
            'network_time_ms': 0,
            'recon_time_ms': 0,
            'distance_time_ms': 0,
            'memory_mb': 0,
            'total_client_time_ms': 0,
            'success': False
        }

        # 记录初始内存
        mem_before = self.process.memory_info().rss / 1024 / 1024  # MB

        # 开始计时
        total_start = time.perf_counter()

        try:
            # 1. 测量 DPF Key Generation
            keygen_start = time.perf_counter()
            keys = self.client.dpf_wrapper.generate_keys('node', node_id)
            keygen_time = (time.perf_counter() - keygen_start) * 1000  # ms

            # 测量 Key Size (平均每个key的大小)
            key_sizes = [len(k) for k in keys]
            avg_key_size = sum(key_sizes) / len(key_sizes) / 1024  # KB

            metrics['key_gen_time_ms'] = keygen_time
            metrics['key_size_kb'] = avg_key_size

            # 2. 发送查询到服务器 (网络时间)
            query_id = f'cost_benchmark_{time.time()}_{node_id}'

            network_start = time.perf_counter()

            # 并行查询所有服务器
            import concurrent.futures
            def query_server(server_id):
                request = {
                    'command': 'query_node_vector',
                    'dpf_key': keys[server_id - 1],
                    'query_id': query_id
                }
                response = self.client._send_request(server_id, request)
                return server_id, response

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.client.connections)) as executor:
                futures = [executor.submit(query_server, sid) for sid in self.client.connections]
                results = {}

                for future in concurrent.futures.as_completed(futures):
                    try:
                        server_id, response = future.result()
                        results[server_id] = response
                    except Exception as e:
                        logger.error(f"查询服务器时出错: {e}")

            network_time = (time.perf_counter() - network_start) * 1000  # ms
            metrics['network_time_ms'] = network_time

            # 检查结果
            successful_responses = {sid: r for sid, r in results.items()
                                  if r and r.get('status') == 'success'}

            if len(successful_responses) < 2:
                logger.warning(f"查询 {node_id} 失败：成功响应的服务器少于2个")
                return metrics

            # 3. 测量 Secret Share Reconstruction
            recon_start = time.perf_counter()
            final_result = self.client._reconstruct_final_result(successful_responses)
            recon_time = (time.perf_counter() - recon_start) * 1000  # ms
            metrics['recon_time_ms'] = recon_time

            # 4. 测量 Distance Computation (余弦相似度)
            distance_start = time.perf_counter()
            similarity = self.client._verify_result(node_id, final_result)
            distance_time = (time.perf_counter() - distance_start) * 1000  # ms
            metrics['distance_time_ms'] = distance_time

            metrics['success'] = True

        except Exception as e:
            logger.error(f"测量查询 {node_id} 时出错: {e}")
            metrics['success'] = False

        # 总时间
        total_time = (time.perf_counter() - total_start) * 1000  # ms
        metrics['total_client_time_ms'] = total_time

        # 记录内存使用 (查询后的内存增量)
        mem_after = self.process.memory_info().rss / 1024 / 1024  # MB
        metrics['memory_mb'] = mem_after - mem_before

        return metrics

    def run_benchmark(self, num_queries: int = 50):
        """运行基准测试"""
        logger.info(f"\n{'='*80}")
        logger.info(f"开始客户端成本测试")
        logger.info(f"数据集: {self.dataset}")
        logger.info(f"查询数量: {num_queries}")
        logger.info(f"{'='*80}\n")

        # 预热
        logger.info("预热查询...")
        for i in range(5):
            node_id = random.randint(0, 9999)
            try:
                self.measure_single_query(node_id)
            except Exception as e:
                logger.warning(f"预热查询 {i+1} 失败: {e}")

        logger.info("预热完成，开始正式测试\n")

        # 正式测试
        for i in range(num_queries):
            node_id = random.randint(0, 9999)

            logger.info(f"测试查询 {i+1}/{num_queries} (node_id={node_id})...")
            metrics = self.measure_single_query(node_id)

            if metrics['success']:
                self.results.append(metrics)
                logger.info(f"  ✓ Key Gen: {metrics['key_gen_time_ms']:.2f}ms, "
                          f"Recon: {metrics['recon_time_ms']:.2f}ms, "
                          f"Distance: {metrics['distance_time_ms']:.2f}ms, "
                          f"Total: {metrics['total_client_time_ms']:.2f}ms")
            else:
                logger.warning(f"  ✗ 查询失败")

            # 每10个查询后等待一下
            if (i + 1) % 10 == 0:
                logger.info(f"已完成 {i+1}/{num_queries} 查询，等待2秒...\n")
                time.sleep(2)

        # 统计结果
        self.print_summary()
        self.save_results()

    def print_summary(self):
        """打印统计摘要"""
        if not self.results:
            logger.error("没有成功的查询结果")
            return

        logger.info(f"\n{'='*80}")
        logger.info("客户端成本分析 - 统计摘要")
        logger.info(f"{'='*80}")
        logger.info(f"数据集: {self.dataset}")
        logger.info(f"成功查询数: {len(self.results)}")
        logger.info(f"{'='*80}\n")

        # 计算统计量
        def calc_stats(values):
            """计算均值和标准差，移除异常值"""
            arr = np.array(values)
            # 移除超过3倍标准差的异常值
            mean = np.mean(arr)
            std = np.std(arr)
            filtered = arr[np.abs(arr - mean) <= 3 * std]

            if len(filtered) == 0:
                filtered = arr

            return {
                'mean': np.mean(filtered),
                'std': np.std(filtered),
                'min': np.min(filtered),
                'max': np.max(filtered),
                'count': len(filtered)
            }

        # 提取各项指标
        key_gen_times = [r['key_gen_time_ms'] for r in self.results]
        key_sizes = [r['key_size_kb'] for r in self.results]
        recon_times = [r['recon_time_ms'] for r in self.results]
        distance_times = [r['distance_time_ms'] for r in self.results]
        network_times = [r['network_time_ms'] for r in self.results]
        total_times = [r['total_client_time_ms'] for r in self.results]
        memories = [r['memory_mb'] for r in self.results]

        # 计算统计量
        key_gen_stats = calc_stats(key_gen_times)
        key_size_stats = calc_stats(key_sizes)
        recon_stats = calc_stats(recon_times)
        distance_stats = calc_stats(distance_times)
        network_stats = calc_stats(network_times)
        total_stats = calc_stats(total_times)
        memory_stats = calc_stats(memories)

        # 打印结果
        logger.info(f"{'指标':<25} {'均值':<15} {'标准差':<15} {'最小值':<15} {'最大值':<15}")
        logger.info(f"{'-'*85}")
        logger.info(f"{'DPF Key Gen (ms)':<25} {key_gen_stats['mean']:<15.3f} {key_gen_stats['std']:<15.3f} {key_gen_stats['min']:<15.3f} {key_gen_stats['max']:<15.3f}")
        logger.info(f"{'DPF Key Size (KB)':<25} {key_size_stats['mean']:<15.3f} {key_size_stats['std']:<15.3f} {key_size_stats['min']:<15.3f} {key_size_stats['max']:<15.3f}")
        logger.info(f"{'Network Time (ms)':<25} {network_stats['mean']:<15.3f} {network_stats['std']:<15.3f} {network_stats['min']:<15.3f} {network_stats['max']:<15.3f}")
        logger.info(f"{'Reconstruction (ms)':<25} {recon_stats['mean']:<15.3f} {recon_stats['std']:<15.3f} {recon_stats['min']:<15.3f} {recon_stats['max']:<15.3f}")
        logger.info(f"{'Distance Comp (ms)':<25} {distance_stats['mean']:<15.3f} {distance_stats['std']:<15.3f} {distance_stats['min']:<15.3f} {distance_stats['max']:<15.3f}")
        logger.info(f"{'Total Client (ms)':<25} {total_stats['mean']:<15.3f} {total_stats['std']:<15.3f} {total_stats['min']:<15.3f} {total_stats['max']:<15.3f}")
        logger.info(f"{'Memory Usage (MB)':<25} {memory_stats['mean']:<15.3f} {memory_stats['std']:<15.3f} {memory_stats['min']:<15.3f} {memory_stats['max']:<15.3f}")
        logger.info(f"{'='*85}\n")

        # 打印表格格式（用于论文）
        logger.info("论文表格格式:")
        logger.info(f"| {self.dataset:<10} | {key_gen_stats['mean']:>6.2f} ± {key_gen_stats['std']:>5.2f} | "
                   f"{key_size_stats['mean']:>8.2f} | {recon_stats['mean']:>6.2f} ± {recon_stats['std']:>5.2f} | "
                   f"{distance_stats['mean']:>6.2f} ± {distance_stats['std']:>5.2f} | "
                   f"{memory_stats['mean']:>7.2f} | {total_stats['mean']:>7.2f} ± {total_stats['std']:>6.2f} |")

        # 存储统计信息
        self.stats = {
            'key_gen': key_gen_stats,
            'key_size': key_size_stats,
            'reconstruction': recon_stats,
            'distance': distance_stats,
            'network': network_stats,
            'total_client': total_stats,
            'memory': memory_stats
        }

    def save_results(self):
        """保存测试结果"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 保存详细结果
        detail_filename = f"client_cost_{self.dataset}_{timestamp}.json"
        with open(detail_filename, 'w') as f:
            json.dump({
                'dataset': self.dataset,
                'timestamp': timestamp,
                'num_queries': len(self.results),
                'statistics': self.stats,
                'raw_results': self.results
            }, f, indent=2)

        logger.info(f"详细结果已保存到: {detail_filename}")

    def cleanup(self):
        """清理资源"""
        self.client.disconnect_from_servers()


def main():
    parser = argparse.ArgumentParser(description='客户端成本分析测试')
    parser.add_argument('--dataset', type=str, default='siftsmall',
                       help='数据集名称 (siftsmall, nfcorpus, laion, tripclick)')
    parser.add_argument('--num-queries', type=int, default=50,
                       help='测试查询数量')

    args = parser.parse_args()

    try:
        benchmark = ClientCostBenchmark(dataset=args.dataset)
        benchmark.run_benchmark(num_queries=args.num_queries)
        benchmark.cleanup()
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()