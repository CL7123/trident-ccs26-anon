#!/usr/bin/env python3
"""
并发性能测试脚本
使用现有的 distributed-deploy/server.py（已支持并发）
只需要客户端并发发送多个查询
"""

import sys
import os
import time
import json
import random
import logging
import argparse
import concurrent.futures
from collections import defaultdict
from typing import List, Dict
import numpy as np

# 添加路径
sys.path.append('~/trident/distributed-deploy')
sys.path.append('~/trident/src')

# 导入现有的客户端
from client import DistributedClient
from config import SERVERS, DEFAULT_DATASET, CONCURRENT_LEVELS, QUERIES_PER_LEVEL

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ConcurrentBenchmark:
    """并发性能测试"""

    def __init__(self, dataset: str = "siftsmall"):
        self.dataset = dataset
        # 不再使用单个共享客户端，每个查询创建独立连接
        self.servers_config = SERVERS
        self.results = defaultdict(list)

    def warmup(self, num_queries: int = 10):
        """预热：发送一些查询让服务器准备好"""
        logger.info(f"预热：发送 {num_queries} 个查询...")

        for i in range(num_queries):
            node_id = random.randint(0, 9999)
            try:
                # 每个查询使用独立的客户端实例
                client = DistributedClient(dataset=self.dataset, servers_config=self.servers_config)
                if not client.connect_to_servers():
                    raise RuntimeError("无法连接到服务器")
                client.test_distributed_query(node_id)
            except Exception as e:
                logger.warning(f"预热查询 {i+1} 失败: {e}")

        logger.info("预热完成")

    def _calculate_concurrency_overlap(self, results: List[Dict]) -> float:
        """
        计算并发重叠度：平均同时执行的查询数量

        算法：
        1. 收集所有查询的时间段
        2. 在每个时间点统计有多少查询在执行
        3. 返回平均值

        返回值：
        - 1.0 = 完全串行（没有重叠）
        - N = 平均有N个查询同时执行
        """
        if not results:
            return 0.0

        # 收集所有事件（开始和结束）
        events = []
        for r in results:
            events.append(('start', r['start_time']))
            events.append(('end', r['end_time']))

        # 按时间排序
        events.sort(key=lambda x: x[1])

        # 计算每个时间段的并发数
        current_concurrent = 0
        total_weighted_concurrent = 0
        total_time = 0
        last_time = events[0][1]

        for event_type, event_time in events:
            if event_time > last_time:
                # 计算上一时间段的贡献
                duration = event_time - last_time
                total_weighted_concurrent += current_concurrent * duration
                total_time += duration
                last_time = event_time

            if event_type == 'start':
                current_concurrent += 1
            else:
                current_concurrent -= 1

        if total_time > 0:
            return total_weighted_concurrent / total_time
        return 0.0

    def query_single(self, query_idx: int) -> Dict:
        """执行单个查询并返回结果"""
        node_id = random.randint(0, 9999)
        query_id = f"benchmark_{time.time()}_{query_idx}"

        try:
            # 每个查询创建独立的客户端实例（独立连接）
            client = DistributedClient(dataset=self.dataset, servers_config=self.servers_config)
            if not client.connect_to_servers():
                raise RuntimeError("无法连接到服务器")

            start_time = time.time()
            result = client.test_distributed_query(node_id)
            end_time = time.time()

            latency = end_time - start_time

            return {
                'success': result is not None,
                'latency': latency,
                'query_id': query_id,
                'node_id': node_id,
                'start_time': start_time,  # 记录开始时间
                'end_time': end_time        # 记录结束时间
            }
        except Exception as e:
            logger.error(f"查询 {query_id} 失败: {e}")
            return {
                'success': False,
                'latency': 0,
                'query_id': query_id,
                'error': str(e),
                'start_time': time.time(),
                'end_time': time.time()
            }

    def test_concurrent_level(self, concurrency: int, num_queries: int) -> Dict:
        """测试特定并发级别"""
        logger.info(f"\n{'='*60}")
        logger.info(f"测试并发级别: {concurrency}")
        logger.info(f"总查询数: {num_queries}")
        logger.info(f"{'='*60}")

        results = []
        start_time = time.time()

        # 使用ThreadPoolExecutor并发执行查询
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            # 提交所有查询
            futures = [executor.submit(self.query_single, i) for i in range(num_queries)]

            # 等待完成并收集结果
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                result = future.result()
                results.append(result)

                # 进度显示
                if (i + 1) % 10 == 0 or (i + 1) == num_queries:
                    logger.info(f"已完成: {i+1}/{num_queries} 查询")

        end_time = time.time()
        total_time = end_time - start_time

        # 统计结果
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]

        if successful:
            latencies = [r['latency'] for r in successful]
            avg_latency = np.mean(latencies)
            p50_latency = np.percentile(latencies, 50)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            throughput = len(successful) / total_time

            # 计算并发重叠度（验证真实并发）
            concurrency_overlap = self._calculate_concurrency_overlap(successful)
        else:
            avg_latency = p50_latency = p95_latency = p99_latency = 0
            throughput = 0
            concurrency_overlap = 0

        summary = {
            'concurrency': concurrency,
            'total_queries': num_queries,
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / num_queries * 100,
            'total_time': total_time,
            'throughput': throughput,
            'avg_latency': avg_latency,
            'p50_latency': p50_latency,
            'p95_latency': p95_latency,
            'p99_latency': p99_latency,
            'all_results': results
        }

        # 打印结果
        logger.info(f"\n结果总结:")
        logger.info(f"  成功查询: {len(successful)}/{num_queries} ({summary['success_rate']:.1f}%)")
        logger.info(f"  失败查询: {len(failed)}")
        logger.info(f"  总耗时: {total_time:.2f}秒")
        logger.info(f"  吞吐量: {throughput:.2f} queries/sec")
        logger.info(f"  平均延迟: {avg_latency:.3f}秒")
        logger.info(f"  P50延迟: {p50_latency:.3f}秒")
        logger.info(f"  P95延迟: {p95_latency:.3f}秒")
        logger.info(f"  P99延迟: {p99_latency:.3f}秒")

        return summary

    def run_benchmark(self, concurrent_levels: List[int], queries_per_level: int):
        """运行完整的基准测试"""
        logger.info("="*80)
        logger.info("并发性能基准测试")
        logger.info(f"数据集: {self.dataset}")
        logger.info(f"并发级别: {concurrent_levels}")
        logger.info(f"每级别查询数: {queries_per_level}")
        logger.info("="*80)

        # 预热
        self.warmup()

        # 测试每个并发级别
        all_summaries = []
        for concurrency in concurrent_levels:
            summary = self.test_concurrent_level(concurrency, queries_per_level)
            all_summaries.append(summary)

            # 等待系统恢复
            logger.info(f"等待5秒让系统恢复...")
            time.sleep(5)

        # 保存结果
        self.save_results(all_summaries)

        # 打印最终总结
        self.print_summary(all_summaries)

        return all_summaries

    def save_results(self, summaries: List[Dict]):
        """保存测试结果到JSON文件"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{self.dataset}_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump({
                'dataset': self.dataset,
                'timestamp': timestamp,
                'summaries': summaries
            }, f, indent=2)

        logger.info(f"\n结果已保存到: {filename}")

    def print_summary(self, summaries: List[Dict]):
        """打印性能总结表格"""
        logger.info("\n" + "="*100)
        logger.info("性能总结")
        logger.info("="*100)
        logger.info(f"{'并发级别':<12} {'成功率':<10} {'吞吐量(qps)':<15} {'平均延迟(s)':<15} {'P95延迟(s)':<15} {'P99延迟(s)':<15}")
        logger.info("-"*100)

        for s in summaries:
            logger.info(
                f"{s['concurrency']:<12} "
                f"{s['success_rate']:<10.1f} "
                f"{s['throughput']:<15.2f} "
                f"{s['avg_latency']:<15.3f} "
                f"{s['p95_latency']:<15.3f} "
                f"{s['p99_latency']:<15.3f}"
            )

        logger.info("="*100)


def main():
    parser = argparse.ArgumentParser(description='并发性能基准测试')
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET,
                       help='数据集名称')
    parser.add_argument('--concurrent-levels', type=str, default='1,2,4,8,16',
                       help='并发级别列表（逗号分隔）')
    parser.add_argument('--queries-per-level', type=int, default=50,
                       help='每个并发级别的查询数量')

    args = parser.parse_args()

    # 解析并发级别
    concurrent_levels = [int(x.strip()) for x in args.concurrent_levels.split(',')]

    # 运行测试
    benchmark = ConcurrentBenchmark(dataset=args.dataset)
    benchmark.run_benchmark(concurrent_levels, args.queries_per_level)


if __name__ == "__main__":
    main()
