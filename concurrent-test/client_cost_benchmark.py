#!/usr/bin/env python3


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

sys.path.append('~/trident/distributed-deploy')
sys.path.append('~/trident/src')

from client import DistributedClient
from config import SERVERS 


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [ClientCost] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ClientCostBenchmark:


    def __init__(self, dataset: str = "siftsmall"):
        self.dataset = dataset
        self.client = DistributedClient(dataset=dataset, servers_config=SERVERS)
        self.process = psutil.Process()
        self.results = []

        if not self.client.connect_to_servers():
            raise RuntimeError("Unable to connect to servers")

        logger.info(f"Client cost test initialized - Dataset: {dataset}")

    def measure_single_query(self, node_id: int) -> Dict:
        """Measure cost metrics for a single query"""
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


        mem_before = self.process.memory_info().rss / 1024 / 1024  # MB


        total_start = time.perf_counter()

        try:
            # 1. Measure DPF Key Generation
            keygen_start = time.perf_counter()
            keys = self.client.dpf_wrapper.generate_keys('node', node_id)
            keygen_time = (time.perf_counter() - keygen_start) * 1000  # ms

            # Measure Key Size (average size per key)
            key_sizes = [len(k) for k in keys]
            avg_key_size = sum(key_sizes) / len(key_sizes) / 1024  # KB

            metrics['key_gen_time_ms'] = keygen_time
            metrics['key_size_kb'] = avg_key_size

            # 2. Send query to servers (network time)
            query_id = f'cost_benchmark_{time.time()}_{node_id}'

            network_start = time.perf_counter()

            # Query all servers in parallel
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
                        logger.error(f"Error querying server: {e}")

            network_time = (time.perf_counter() - network_start) * 1000  # ms
            metrics['network_time_ms'] = network_time

            # Check results
            successful_responses = {sid: r for sid, r in results.items()
                                  if r and r.get('status') == 'success'}

            if len(successful_responses) < 2:
                logger.warning(f"Query {node_id} failed: fewer than 2 servers responded successfully")
                return metrics

            # 3. Measure Secret Share Reconstruction
            recon_start = time.perf_counter()
            final_result = self.client._reconstruct_final_result(successful_responses)
            recon_time = (time.perf_counter() - recon_start) * 1000  # ms
            metrics['recon_time_ms'] = recon_time

            # 4. Measure Distance Computation (cosine similarity)
            distance_start = time.perf_counter()
            similarity = self.client._verify_result(node_id, final_result)
            distance_time = (time.perf_counter() - distance_start) * 1000  # ms
            metrics['distance_time_ms'] = distance_time

            metrics['success'] = True

        except Exception as e:
            logger.error(f"Error measuring query {node_id}: {e}")
            metrics['success'] = False

        # Total time
        total_time = (time.perf_counter() - total_start) * 1000  # ms
        metrics['total_client_time_ms'] = total_time

        # Record memory usage (memory increase after query)
        mem_after = self.process.memory_info().rss / 1024 / 1024  # MB
        metrics['memory_mb'] = mem_after - mem_before

        return metrics

    def run_benchmark(self, num_queries: int = 50):
        """Run benchmark test"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting client cost test")
        logger.info(f"Dataset: {self.dataset}")
        logger.info(f"Number of queries: {num_queries}")
        logger.info(f"{'='*80}\n")

        # Warmup
        logger.info("Warmup queries...")
        for i in range(5):
            node_id = random.randint(0, 9999)
            try:
                self.measure_single_query(node_id)
            except Exception as e:
                logger.warning(f"Warmup query {i+1} failed: {e}")

        logger.info("Warmup completed, starting formal test\n")

        # Formal test
        for i in range(num_queries):
            node_id = random.randint(0, 9999)

            logger.info(f"Test query {i+1}/{num_queries} (node_id={node_id})...")
            metrics = self.measure_single_query(node_id)

            if metrics['success']:
                self.results.append(metrics)
                logger.info(f"  ✓ Key Gen: {metrics['key_gen_time_ms']:.2f}ms, "
                          f"Recon: {metrics['recon_time_ms']:.2f}ms, "
                          f"Distance: {metrics['distance_time_ms']:.2f}ms, "
                          f"Total: {metrics['total_client_time_ms']:.2f}ms")
            else:
                logger.warning(f"  ✗ Query failed")

            # Wait after every 10 queries
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i+1}/{num_queries} queries, waiting 2 seconds...\n")
                time.sleep(2)

        # Aggregate results
        self.print_summary()
        self.save_results()

    def print_summary(self):
        """Print statistical summary"""
        if not self.results:
            logger.error("No successful query results")
            return

        logger.info(f"\n{'='*80}")
        logger.info("Client Cost Analysis - Statistical Summary")
        logger.info(f"{'='*80}")
        logger.info(f"Dataset: {self.dataset}")
        logger.info(f"Successful queries: {len(self.results)}")
        logger.info(f"{'='*80}\n")

        # Calculate statistics
        def calc_stats(values):
            """Calculate mean and standard deviation, remove outliers"""
            arr = np.array(values)
            # Remove outliers exceeding 3 standard deviations
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

        # Extract metrics
        key_gen_times = [r['key_gen_time_ms'] for r in self.results]
        key_sizes = [r['key_size_kb'] for r in self.results]
        recon_times = [r['recon_time_ms'] for r in self.results]
        distance_times = [r['distance_time_ms'] for r in self.results]
        network_times = [r['network_time_ms'] for r in self.results]
        total_times = [r['total_client_time_ms'] for r in self.results]
        memories = [r['memory_mb'] for r in self.results]

        # Compute statistics
        key_gen_stats = calc_stats(key_gen_times)
        key_size_stats = calc_stats(key_sizes)
        recon_stats = calc_stats(recon_times)
        distance_stats = calc_stats(distance_times)
        network_stats = calc_stats(network_times)
        total_stats = calc_stats(total_times)
        memory_stats = calc_stats(memories)

        # Print results
        logger.info(f"{'Metric':<25} {'Mean':<15} {'Std Dev':<15} {'Min':<15} {'Max':<15}")
        logger.info(f"{'-'*85}")
        logger.info(f"{'DPF Key Gen (ms)':<25} {key_gen_stats['mean']:<15.3f} {key_gen_stats['std']:<15.3f} {key_gen_stats['min']:<15.3f} {key_gen_stats['max']:<15.3f}")
        logger.info(f"{'DPF Key Size (KB)':<25} {key_size_stats['mean']:<15.3f} {key_size_stats['std']:<15.3f} {key_size_stats['min']:<15.3f} {key_size_stats['max']:<15.3f}")
        logger.info(f"{'Network Time (ms)':<25} {network_stats['mean']:<15.3f} {network_stats['std']:<15.3f} {network_stats['min']:<15.3f} {network_stats['max']:<15.3f}")
        logger.info(f"{'Reconstruction (ms)':<25} {recon_stats['mean']:<15.3f} {recon_stats['std']:<15.3f} {recon_stats['min']:<15.3f} {recon_stats['max']:<15.3f}")
        logger.info(f"{'Distance Comp (ms)':<25} {distance_stats['mean']:<15.3f} {distance_stats['std']:<15.3f} {distance_stats['min']:<15.3f} {distance_stats['max']:<15.3f}")
        logger.info(f"{'Total Client (ms)':<25} {total_stats['mean']:<15.3f} {total_stats['std']:<15.3f} {total_stats['min']:<15.3f} {total_stats['max']:<15.3f}")
        logger.info(f"{'Memory Usage (MB)':<25} {memory_stats['mean']:<15.3f} {memory_stats['std']:<15.3f} {memory_stats['min']:<15.3f} {memory_stats['max']:<15.3f}")
        logger.info(f"{'='*85}\n")

        # Print table format (for paper)
        logger.info("Paper table format:")
        logger.info(f"| {self.dataset:<10} | {key_gen_stats['mean']:>6.2f} ± {key_gen_stats['std']:>5.2f} | "
                   f"{key_size_stats['mean']:>8.2f} | {recon_stats['mean']:>6.2f} ± {recon_stats['std']:>5.2f} | "
                   f"{distance_stats['mean']:>6.2f} ± {distance_stats['std']:>5.2f} | "
                   f"{memory_stats['mean']:>7.2f} | {total_stats['mean']:>7.2f} ± {total_stats['std']:>6.2f} |")

        # Store statistics
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
        """Save test results"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        detail_filename = f"client_cost_{self.dataset}_{timestamp}.json"
        with open(detail_filename, 'w') as f:
            json.dump({
                'dataset': self.dataset,
                'timestamp': timestamp,
                'num_queries': len(self.results),
                'statistics': self.stats,
                'raw_results': self.results
            }, f, indent=2)

        logger.info(f"Detailed results saved to: {detail_filename}")

    def cleanup(self):
        """Clean up resources"""
        self.client.disconnect_from_servers()


def main():
    parser = argparse.ArgumentParser(description='Client cost analysis test')
    parser.add_argument('--dataset', type=str, default='siftsmall',
                       help='Dataset name (siftsmall, nfcorpus, laion, tripclick)')
    parser.add_argument('--num-queries', type=int, default=50,
                       help='Number of test queries')

    args = parser.parse_args()

    try:
        benchmark = ClientCostBenchmark(dataset=args.dataset)
        benchmark.run_benchmark(num_queries=args.num_queries)
        benchmark.cleanup()
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()