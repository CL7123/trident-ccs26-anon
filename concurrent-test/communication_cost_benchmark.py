#!/usr/bin/env python3
"""
通信成本分析测试脚本
测量 Client-Server 通信开销（大小和时间）
计算 Server-Server MPC 通信量（理论值）
"""

import sys
import os
import time
import json
import random
import logging
import argparse
import numpy as np
from typing import Dict, List
import concurrent.futures

# 添加路径
sys.path.append('~/trident/distributed-deploy')
sys.path.append('~/trident/src')

from client import DistributedClient
from config import SERVERS
from domain_config import get_config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [CommCost] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class CommunicationCostBenchmark:
    """通信成本测试"""

    def __init__(self, dataset: str = "siftsmall"):
        self.dataset = dataset
        self.config = get_config(dataset)
        self.client = DistributedClient(dataset=dataset, servers_config=SERVERS)
        self.results = []

        # 计算MPC通信量（理论值）
        self.mpc_size_mb = self._calculate_mpc_exchange_size()

        # 连接到服务器
        if not self.client.connect_to_servers():
            raise RuntimeError("无法连接到服务器")

        logger.info(f"通信成本测试初始化完成 - 数据集: {dataset}")
        logger.info(f"MPC交换数据量（理论值）: {self.mpc_size_mb:.2f} MB/query")

    def _calculate_mpc_exchange_size(self) -> float:
        """
        计算Server间MPC通信量（理论值）

        Phase 3 中每个server交换 e_shares 和 f_shares:
        - e_shares: int32[num_nodes]
        - f_shares: int32[num_nodes]
        - 每个server发送给另外2个servers
        - 3个servers总通信量
        """
        num_nodes = self.config.num_docs

        # 每个share的大小（int32 = 4 bytes）
        share_size_bytes = num_nodes * 4

        # 每个server发送: (e_shares + f_shares) × 2个目标servers
        per_server_send_bytes = share_size_bytes * 2 * 2

        # 3个servers总通信量
        total_bytes = per_server_send_bytes * 3

        # 转换为MB
        return total_bytes / (1024 ** 2)

    def measure_single_query(self, node_id: int) -> Dict:
        """测量单个查询的通信成本"""
        metrics = {
            'node_id': node_id,
            'upload_size_mb': 0,
            'download_size_mb': 0,
            'upload_time_s': 0,
            'download_time_s': 0,
            'comm_time_s': 0,
            'phase1_time_s': 0,
            'phase2_time_s': 0,
            'phase3_time_s': 0,
            'phase4_time_s': 0,
            'comp_time_s': 0,
            'total_time_s': 0,
            'comm_percentage': 0,
            'mpc_size_mb': self.mpc_size_mb,
            'success': False
        }

        try:
            # 1. 生成DPF keys并测量大小
            keys = self.client.dpf_wrapper.generate_keys('node', node_id)
            upload_size_bytes = sum(len(k) for k in keys)
            metrics['upload_size_mb'] = upload_size_bytes / (1024 ** 2)

            # 2. 生成query_id
            query_id = f'comm_benchmark_{time.time()}_{node_id}'

            # 开始计时总时间
            total_start = time.perf_counter()

            # 3. 测量上传时间（发送queries到所有servers）
            upload_start = time.perf_counter()

            def query_server(server_id):
                request = {
                    'command': 'query_node_vector',
                    'dpf_key': keys[server_id - 1],
                    'query_id': query_id
                }
                # 这里的_send_request会发送数据
                response = self.client._send_request(server_id, request)
                return server_id, response

            # 并行发送到所有servers
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.client.connections)) as executor:
                futures = [executor.submit(query_server, sid) for sid in self.client.connections]

                # 上传时间 = 所有请求发送完成的时间
                # 这里我们等待第一个response来确保所有请求都已发送
                results = {}
                first_response_received = False

                for future in concurrent.futures.as_completed(futures):
                    try:
                        server_id, response = future.result()
                        results[server_id] = response

                        if not first_response_received:
                            # 第一个response收到，说明上传已完成
                            upload_time = time.perf_counter() - upload_start
                            metrics['upload_time_s'] = upload_time
                            first_response_received = True

                            # 开始测量下载时间
                            download_start = time.perf_counter()

                    except Exception as e:
                        logger.error(f"查询服务器时出错: {e}")

            # 4. 下载时间 = 最后一个response收到的时间 - 第一个response收到的时间
            download_time = time.perf_counter() - download_start
            metrics['download_time_s'] = download_time

            # 5. 计算response大小
            successful_responses = {sid: r for sid, r in results.items()
                                  if r and r.get('status') == 'success'}

            if len(successful_responses) < 2:
                logger.warning(f"查询 {node_id} 失败：成功响应的服务器少于2个")
                return metrics

            # 计算download大小（JSON序列化的response）
            download_size_bytes = sum(len(json.dumps(r).encode())
                                     for r in successful_responses.values())
            metrics['download_size_mb'] = download_size_bytes / (1024 ** 2)

            # 6. 提取server端的timing信息
            for server_id, result in successful_responses.items():
                timing = result.get('timing', {})
                metrics['phase1_time_s'] = timing.get('phase1_time', 0) / 1000
                metrics['phase2_time_s'] = timing.get('phase2_time', 0) / 1000
                metrics['phase3_time_s'] = timing.get('phase3_time', 0) / 1000
                metrics['phase4_time_s'] = timing.get('phase4_time', 0) / 1000
                break  # 只需要一个server的timing

            # 7. 计算综合指标
            total_time = time.perf_counter() - total_start
            metrics['total_time_s'] = total_time

            # 通信时间 = upload + download
            comm_time = metrics['upload_time_s'] + metrics['download_time_s']
            metrics['comm_time_s'] = comm_time

            # 计算时间 = phase1 + phase2 + phase4 (phase3是通信)
            comp_time = (metrics['phase1_time_s'] +
                        metrics['phase2_time_s'] +
                        metrics['phase4_time_s'])
            metrics['comp_time_s'] = comp_time

            # 通信占比
            if total_time > 0:
                metrics['comm_percentage'] = (comm_time / total_time) * 100

            metrics['success'] = True

        except Exception as e:
            logger.error(f"测量查询 {node_id} 时出错: {e}")
            import traceback
            traceback.print_exc()
            metrics['success'] = False

        return metrics

    def run_benchmark(self, num_queries: int = 50):
        """运行基准测试"""
        logger.info(f"\n{'='*80}")
        logger.info(f"开始通信成本测试")
        logger.info(f"数据集: {self.dataset}")
        logger.info(f"查询数量: {num_queries}")
        logger.info(f"文档数量: {self.config.num_docs:,}")
        logger.info(f"MPC交换数据量（理论值）: {self.mpc_size_mb:.2f} MB/query")
        logger.info(f"{'='*80}\n")

        # 预热
        logger.info("预热查询...")
        for i in range(5):
            node_id = random.randint(0, min(9999, self.config.num_docs - 1))
            try:
                self.measure_single_query(node_id)
            except Exception as e:
                logger.warning(f"预热查询 {i+1} 失败: {e}")

        logger.info("预热完成，开始正式测试\n")

        # 正式测试
        for i in range(num_queries):
            node_id = random.randint(0, min(9999, self.config.num_docs - 1))

            logger.info(f"测试查询 {i+1}/{num_queries} (node_id={node_id})...")
            metrics = self.measure_single_query(node_id)

            if metrics['success']:
                self.results.append(metrics)
                logger.info(f"  ✓ Upload: {metrics['upload_size_mb']:.3f}MB/{metrics['upload_time_s']:.3f}s, "
                          f"Download: {metrics['download_size_mb']:.3f}MB/{metrics['download_time_s']:.3f}s, "
                          f"Comm%: {metrics['comm_percentage']:.1f}%")
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

        logger.info(f"\n{'='*100}")
        logger.info("通信成本分析 - 统计摘要")
        logger.info(f"{'='*100}")
        logger.info(f"数据集: {self.dataset}")
        logger.info(f"成功查询数: {len(self.results)}")
        logger.info(f"{'='*100}\n")

        # 计算统计量
        def calc_stats(values):
            """计算均值和标准差，移除异常值"""
            arr = np.array(values)
            mean = np.mean(arr)
            std = np.std(arr)
            # 移除超过3倍标准差的异常值
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
        upload_sizes = [r['upload_size_mb'] for r in self.results]
        download_sizes = [r['download_size_mb'] for r in self.results]
        upload_times = [r['upload_time_s'] for r in self.results]
        download_times = [r['download_time_s'] for r in self.results]
        comm_times = [r['comm_time_s'] for r in self.results]
        comp_times = [r['comp_time_s'] for r in self.results]
        total_times = [r['total_time_s'] for r in self.results]
        comm_percentages = [r['comm_percentage'] for r in self.results]

        # 计算统计量
        upload_size_stats = calc_stats(upload_sizes)
        download_size_stats = calc_stats(download_sizes)
        upload_time_stats = calc_stats(upload_times)
        download_time_stats = calc_stats(download_times)
        comm_time_stats = calc_stats(comm_times)
        comp_time_stats = calc_stats(comp_times)
        total_time_stats = calc_stats(total_times)
        comm_pct_stats = calc_stats(comm_percentages)

        # MPC size是固定值
        mpc_size = self.mpc_size_mb

        # 打印结果表格
        logger.info(f"{'指标':<30} {'均值':<15} {'标准差':<15} {'最小值':<15} {'最大值':<15}")
        logger.info(f"{'-'*90}")
        logger.info(f"{'Upload Size (MB)':<30} {upload_size_stats['mean']:<15.4f} {upload_size_stats['std']:<15.4f} {upload_size_stats['min']:<15.4f} {upload_size_stats['max']:<15.4f}")
        logger.info(f"{'Download Size (MB)':<30} {download_size_stats['mean']:<15.4f} {download_size_stats['std']:<15.4f} {download_size_stats['min']:<15.4f} {download_size_stats['max']:<15.4f}")
        logger.info(f"{'MPC Exchange Size (MB/query)':<30} {mpc_size:<15.4f} {'(fixed)':<15} {'-':<15} {'-':<15}")
        logger.info(f"{'Upload Time (s)':<30} {upload_time_stats['mean']:<15.4f} {upload_time_stats['std']:<15.4f} {upload_time_stats['min']:<15.4f} {upload_time_stats['max']:<15.4f}")
        logger.info(f"{'Download Time (s)':<30} {download_time_stats['mean']:<15.4f} {download_time_stats['std']:<15.4f} {download_time_stats['min']:<15.4f} {download_time_stats['max']:<15.4f}")
        logger.info(f"{'Client-Server Comm Time (s)':<30} {comm_time_stats['mean']:<15.4f} {comm_time_stats['std']:<15.4f} {comm_time_stats['min']:<15.4f} {comm_time_stats['max']:<15.4f}")
        logger.info(f"{'Computation Time (s)':<30} {comp_time_stats['mean']:<15.4f} {comp_time_stats['std']:<15.4f} {comp_time_stats['min']:<15.4f} {comp_time_stats['max']:<15.4f}")
        logger.info(f"{'Total Query Time (s)':<30} {total_time_stats['mean']:<15.4f} {total_time_stats['std']:<15.4f} {total_time_stats['min']:<15.4f} {total_time_stats['max']:<15.4f}")
        logger.info(f"{'Communication Percentage (%)':<30} {comm_pct_stats['mean']:<15.2f} {comm_pct_stats['std']:<15.2f} {comm_pct_stats['min']:<15.2f} {comm_pct_stats['max']:<15.2f}")
        logger.info(f"{'='*90}\n")

        # 打印论文表格格式
        logger.info("论文表格格式:")
        logger.info(f"| {self.dataset:<10} | "
                   f"{upload_size_stats['mean']:>5.2f} ± {upload_size_stats['std']:>4.2f} | "
                   f"{download_size_stats['mean']:>5.2f} ± {download_size_stats['std']:>4.2f} | "
                   f"{comm_time_stats['mean']:>5.2f} ± {comm_time_stats['std']:>4.2f} | "
                   f"{mpc_size:>8.2f} | "
                   f"{comp_time_stats['mean']:>5.2f} ± {comp_time_stats['std']:>4.2f} | "
                   f"{total_time_stats['mean']:>5.1f} ± {total_time_stats['std']:>4.1f} | "
                   f"{comm_pct_stats['mean']:>5.1f}% |")

        # 存储统计信息
        self.stats = {
            'upload_size': upload_size_stats,
            'download_size': download_size_stats,
            'mpc_size': mpc_size,
            'upload_time': upload_time_stats,
            'download_time': download_time_stats,
            'comm_time': comm_time_stats,
            'comp_time': comp_time_stats,
            'total_time': total_time_stats,
            'comm_percentage': comm_pct_stats
        }

    def save_results(self):
        """保存测试结果"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 保存详细结果
        detail_filename = f"comm_cost_{self.dataset}_{timestamp}.json"
        with open(detail_filename, 'w') as f:
            json.dump({
                'dataset': self.dataset,
                'timestamp': timestamp,
                'num_queries': len(self.results),
                'num_docs': self.config.num_docs,
                'mpc_size_mb': self.mpc_size_mb,
                'statistics': self.stats,
                'raw_results': self.results
            }, f, indent=2)

        logger.info(f"详细结果已保存到: {detail_filename}")

    def cleanup(self):
        """清理资源"""
        self.client.disconnect_from_servers()


def main():
    parser = argparse.ArgumentParser(description='通信成本分析测试')
    parser.add_argument('--dataset', type=str, default='siftsmall',
                       help='数据集名称 (siftsmall, nfcorpus, laion, tripclick)')
    parser.add_argument('--num-queries', type=int, default=50,
                       help='测试查询数量')

    args = parser.parse_args()

    try:
        benchmark = CommunicationCostBenchmark(dataset=args.dataset)
        benchmark.run_benchmark(num_queries=args.num_queries)
        benchmark.cleanup()
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
