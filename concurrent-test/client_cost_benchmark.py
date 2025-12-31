#!/usr/bin/env python3
"""
[CN]
[CN] DPF key [CN]、[CN]、[CN]calculate[CN]
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

# [CN]
sys.path.append('~/trident/distributed-deploy')
sys.path.append('~/trident/src')

from client import DistributedClient
from config import SERVERS  # [CN]（[CN]IP）

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [ClientCost] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ClientCostBenchmark:
    """[CN]"""

    def __init__(self, dataset: str = "siftsmall"):
        self.dataset = dataset
        self.client = DistributedClient(dataset=dataset, servers_config=SERVERS)
        self.process = psutil.Process()
        self.results = []

        # connect[CN]
        if not self.client.connect_to_servers():
            raise RuntimeError("[CN]connect[CN]")

        logger.info(f"[CN]initialize[CN] - Dataset: {dataset}")

    def measure_single_query(self, node_id: int) -> Dict:
        """[CN]"""
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

        # [CN]
        mem_before = self.process.memory_info().rss / 1024 / 1024  # MB

        # [CN]
        total_start = time.perf_counter()

        try:
            # 1. [CN] DPF Key Generation
            keygen_start = time.perf_counter()
            keys = self.client.dpf_wrapper.generate_keys('node', node_id)
            keygen_time = (time.perf_counter() - keygen_start) * 1000  # ms

            # [CN] Key Size ([CN]key[CN])
            key_sizes = [len(k) for k in keys]
            avg_key_size = sum(key_sizes) / len(key_sizes) / 1024  # KB

            metrics['key_gen_time_ms'] = keygen_time
            metrics['key_size_kb'] = avg_key_size

            # 2. send[CN] ([CN])
            query_id = f'cost_benchmark_{time.time()}_{node_id}'

            network_start = time.perf_counter()

            # [CN]
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
                        logger.error(f"[CN]: {e}")

            network_time = (time.perf_counter() - network_start) * 1000  # ms
            metrics['network_time_ms'] = network_time

            # [CN]
            successful_responses = {sid: r for sid, r in results.items()
                                  if r and r.get('status') == 'success'}

            if len(successful_responses) < 2:
                logger.warning(f"[CN] {node_id} [CN]：[CN]2[CN]")
                return metrics

            # 3. [CN] Secret Share Reconstruction
            recon_start = time.perf_counter()
            final_result = self.client._reconstruct_final_result(successful_responses)
            recon_time = (time.perf_counter() - recon_start) * 1000  # ms
            metrics['recon_time_ms'] = recon_time

            # 4. [CN] Distance Computation ([CN])
            distance_start = time.perf_counter()
            similarity = self.client._verify_result(node_id, final_result)
            distance_time = (time.perf_counter() - distance_start) * 1000  # ms
            metrics['distance_time_ms'] = distance_time

            metrics['success'] = True

        except Exception as e:
            logger.error(f"[CN] {node_id} [CN]: {e}")
            metrics['success'] = False

        # [CN]
        total_time = (time.perf_counter() - total_start) * 1000  # ms
        metrics['total_client_time_ms'] = total_time

        # [CN] ([CN])
        mem_after = self.process.memory_info().rss / 1024 / 1024  # MB
        metrics['memory_mb'] = mem_after - mem_before

        return metrics

    def run_benchmark(self, num_queries: int = 50):
        """[CN]"""
        logger.info(f"\n{'='*80}")
        logger.info(f"[CN]")
        logger.info(f"Dataset: {self.dataset}")
        logger.info(f"Number of queries[CN]: {num_queries}")
        logger.info(f"{'='*80}\n")

        # [CN]
        logger.info("[CN]...")
        for i in range(5):
            node_id = random.randint(0, 9999)
            try:
                self.measure_single_query(node_id)
            except Exception as e:
                logger.warning(f"[CN] {i+1} [CN]: {e}")

        logger.info("[CN]，[CN]\n")

        # [CN]
        for i in range(num_queries):
            node_id = random.randint(0, 9999)

            logger.info(f"[CN] {i+1}/{num_queries} (node_id={node_id})...")
            metrics = self.measure_single_query(node_id)

            if metrics['success']:
                self.results.append(metrics)
                logger.info(f"  ✓ Key Gen: {metrics['key_gen_time_ms']:.2f}ms, "
                          f"Recon: {metrics['recon_time_ms']:.2f}ms, "
                          f"Distance: {metrics['distance_time_ms']:.2f}ms, "
                          f"Total: {metrics['total_client_time_ms']:.2f}ms")
            else:
                logger.warning(f"  ✗ [CN]")

            # [CN]10[CN]
            if (i + 1) % 10 == 0:
                logger.info(f"[CN] {i+1}/{num_queries} [CN]，[CN]2[CN]...\n")
                time.sleep(2)

        # [CN]
        self.print_summary()
        self.save_results()

    def print_summary(self):
        """print[CN]"""
        if not self.results:
            logger.error("[CN]")
            return

        logger.info(f"\n{'='*80}")
        logger.info("[CN] - [CN]")
        logger.info(f"{'='*80}")
        logger.info(f"Dataset: {self.dataset}")
        logger.info(f"[CN]Number of queries: {len(self.results)}")
        logger.info(f"{'='*80}\n")

        # calculate[CN]
        def calc_stats(values):
            """calculate[CN]，[CN]"""
            arr = np.array(values)
            # [CN]3[CN]
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

        # [CN]
        key_gen_times = [r['key_gen_time_ms'] for r in self.results]
        key_sizes = [r['key_size_kb'] for r in self.results]
        recon_times = [r['recon_time_ms'] for r in self.results]
        distance_times = [r['distance_time_ms'] for r in self.results]
        network_times = [r['network_time_ms'] for r in self.results]
        total_times = [r['total_client_time_ms'] for r in self.results]
        memories = [r['memory_mb'] for r in self.results]

        # calculate[CN]
        key_gen_stats = calc_stats(key_gen_times)
        key_size_stats = calc_stats(key_sizes)
        recon_stats = calc_stats(recon_times)
        distance_stats = calc_stats(distance_times)
        network_stats = calc_stats(network_times)
        total_stats = calc_stats(total_times)
        memory_stats = calc_stats(memories)

        # print[CN]
        logger.info(f"{'[CN]':<25} {'[CN]':<15} {'[CN]':<15} {'[CN]':<15} {'[CN]':<15}")
        logger.info(f"{'-'*85}")
        logger.info(f"{'DPF Key Gen (ms)':<25} {key_gen_stats['mean']:<15.3f} {key_gen_stats['std']:<15.3f} {key_gen_stats['min']:<15.3f} {key_gen_stats['max']:<15.3f}")
        logger.info(f"{'DPF Key Size (KB)':<25} {key_size_stats['mean']:<15.3f} {key_size_stats['std']:<15.3f} {key_size_stats['min']:<15.3f} {key_size_stats['max']:<15.3f}")
        logger.info(f"{'Network Time (ms)':<25} {network_stats['mean']:<15.3f} {network_stats['std']:<15.3f} {network_stats['min']:<15.3f} {network_stats['max']:<15.3f}")
        logger.info(f"{'Reconstruction (ms)':<25} {recon_stats['mean']:<15.3f} {recon_stats['std']:<15.3f} {recon_stats['min']:<15.3f} {recon_stats['max']:<15.3f}")
        logger.info(f"{'Distance Comp (ms)':<25} {distance_stats['mean']:<15.3f} {distance_stats['std']:<15.3f} {distance_stats['min']:<15.3f} {distance_stats['max']:<15.3f}")
        logger.info(f"{'Total Client (ms)':<25} {total_stats['mean']:<15.3f} {total_stats['std']:<15.3f} {total_stats['min']:<15.3f} {total_stats['max']:<15.3f}")
        logger.info(f"{'Memory Usage (MB)':<25} {memory_stats['mean']:<15.3f} {memory_stats['std']:<15.3f} {memory_stats['min']:<15.3f} {memory_stats['max']:<15.3f}")
        logger.info(f"{'='*85}\n")

        # print[CN]（[CN]）
        logger.info("[CN]:")
        logger.info(f"| {self.dataset:<10} | {key_gen_stats['mean']:>6.2f} ± {key_gen_stats['std']:>5.2f} | "
                   f"{key_size_stats['mean']:>8.2f} | {recon_stats['mean']:>6.2f} ± {recon_stats['std']:>5.2f} | "
                   f"{distance_stats['mean']:>6.2f} ± {distance_stats['std']:>5.2f} | "
                   f"{memory_stats['mean']:>7.2f} | {total_stats['mean']:>7.2f} ± {total_stats['std']:>6.2f} |")

        # [CN]
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
        """[CN]Test results"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # [CN]
        detail_filename = f"client_cost_{self.dataset}_{timestamp}.json"
        with open(detail_filename, 'w') as f:
            json.dump({
                'dataset': self.dataset,
                'timestamp': timestamp,
                'num_queries': len(self.results),
                'statistics': self.stats,
                'raw_results': self.results
            }, f, indent=2)

        logger.info(f"[CN]: {detail_filename}")

    def cleanup(self):
        """[CN]"""
        self.client.disconnect_from_servers()


def main():
    parser = argparse.ArgumentParser(description='[CN]')
    parser.add_argument('--dataset', type=str, default='siftsmall',
                       help='Dataset[CN] (siftsmall, nfcorpus, laion, tripclick)')
    parser.add_argument('--num-queries', type=int, default=50,
                       help='[CN]Number of queries[CN]')

    args = parser.parse_args()

    try:
        benchmark = ClientCostBenchmark(dataset=args.dataset)
        benchmark.run_benchmark(num_queries=args.num_queries)
        benchmark.cleanup()
    except Exception as e:
        logger.error(f"[CN]: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()