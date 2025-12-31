#!/usr/bin/env python3
"""
[CN]
[CN] distributed-deploy/server.py（[CN]）
[CN]send[CN]
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

# [CN]
sys.path.append('~/trident/distributed-deploy')
sys.path.append('~/trident/src')

# [CN]
from client import DistributedClient
from config import SERVERS, DEFAULT_DATASET, CONCURRENT_LEVELS, QUERIES_PER_LEVEL

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ConcurrentBenchmark:
    """[CN]"""

    def __init__(self, dataset: str = "siftsmall"):
        self.dataset = dataset
        # [CN]，[CN]create[CN]connect
        self.servers_config = SERVERS
        self.results = defaultdict(list)

    def warmup(self, num_queries: int = 10):
        """[CN]：send[CN]"""
        logger.info(f"[CN]：send {num_queries} [CN]...")

        for i in range(num_queries):
            node_id = random.randint(0, 9999)
            try:
                # [CN]instance
                client = DistributedClient(dataset=self.dataset, servers_config=self.servers_config)
                if not client.connect_to_servers():
                    raise RuntimeError("[CN]connect[CN]")
                client.test_distributed_query(node_id)
            except Exception as e:
                logger.warning(f"[CN] {i+1} [CN]: {e}")

        logger.info("[CN]")

    def _calculate_concurrency_overlap(self, results: List[Dict]) -> float:
        """
        calculate[CN]：[CN]Number of queries[CN]

        [CN]：
        1. [CN]
        2. [CN]
        3. return[CN]

        return[CN]：
        - 1.0 = [CN]（[CN]）
        - N = [CN]N[CN]
        """
        if not results:
            return 0.0

        # [CN]（[CN]）
        events = []
        for r in results:
            events.append(('start', r['start_time']))
            events.append(('end', r['end_time']))

        # [CN]
        events.sort(key=lambda x: x[1])

        # calculate[CN]
        current_concurrent = 0
        total_weighted_concurrent = 0
        total_time = 0
        last_time = events[0][1]

        for event_type, event_time in events:
            if event_time > last_time:
                # calculate[CN]
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
        """[CN]return[CN]"""
        node_id = random.randint(0, 9999)
        query_id = f"benchmark_{time.time()}_{query_idx}"

        try:
            # [CN]create[CN]instance（[CN]connect）
            client = DistributedClient(dataset=self.dataset, servers_config=self.servers_config)
            if not client.connect_to_servers():
                raise RuntimeError("[CN]connect[CN]")

            start_time = time.time()
            result = client.test_distributed_query(node_id)
            end_time = time.time()

            latency = end_time - start_time

            return {
                'success': result is not None,
                'latency': latency,
                'query_id': query_id,
                'node_id': node_id,
                'start_time': start_time,  # [CN]
                'end_time': end_time        # [CN]
            }
        except Exception as e:
            logger.error(f"[CN] {query_id} [CN]: {e}")
            return {
                'success': False,
                'latency': 0,
                'query_id': query_id,
                'error': str(e),
                'start_time': time.time(),
                'end_time': time.time()
            }

    def test_concurrent_level(self, concurrency: int, num_queries: int) -> Dict:
        """[CN]"""
        logger.info(f"\n{'='*60}")
        logger.info(f"[CN]: {concurrency}")
        logger.info(f"[CN]Number of queries: {num_queries}")
        logger.info(f"{'='*60}")

        results = []
        start_time = time.time()

        # [CN]ThreadPoolExecutor[CN]
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            # [CN]
            futures = [executor.submit(self.query_single, i) for i in range(num_queries)]

            # [CN]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                result = future.result()
                results.append(result)

                # [CN]
                if (i + 1) % 10 == 0 or (i + 1) == num_queries:
                    logger.info(f"[CN]: {i+1}/{num_queries} [CN]")

        end_time = time.time()
        total_time = end_time - start_time

        # [CN]
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]

        if successful:
            latencies = [r['latency'] for r in successful]
            avg_latency = np.mean(latencies)
            p50_latency = np.percentile(latencies, 50)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            throughput = len(successful) / total_time

            # calculate[CN]（[CN]）
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

        # print[CN]
        logger.info(f"\n[CN]:")
        logger.info(f"  [CN]: {len(successful)}/{num_queries} ({summary['success_rate']:.1f}%)")
        logger.info(f"  [CN]: {len(failed)}")
        logger.info(f"  [CN]: {total_time:.2f}[CN]")
        logger.info(f"  [CN]: {throughput:.2f} queries/sec")
        logger.info(f"  Average latency: {avg_latency:.3f}[CN]")
        logger.info(f"  P50[CN]: {p50_latency:.3f}[CN]")
        logger.info(f"  P95[CN]: {p95_latency:.3f}[CN]")
        logger.info(f"  P99[CN]: {p99_latency:.3f}[CN]")

        return summary

    def run_benchmark(self, concurrent_levels: List[int], queries_per_level: int):
        """[CN]"""
        logger.info("="*80)
        logger.info("[CN]")
        logger.info(f"Dataset: {self.dataset}")
        logger.info(f"[CN]: {concurrent_levels}")
        logger.info(f"[CN]Number of queries: {queries_per_level}")
        logger.info("="*80)

        # [CN]
        self.warmup()

        # [CN]
        all_summaries = []
        for concurrency in concurrent_levels:
            summary = self.test_concurrent_level(concurrency, queries_per_level)
            all_summaries.append(summary)

            # [CN]
            logger.info(f"[CN]5[CN]...")
            time.sleep(5)

        # [CN]
        self.save_results(all_summaries)

        # print[CN]
        self.print_summary(all_summaries)

        return all_summaries

    def save_results(self, summaries: List[Dict]):
        """[CN]Test results[CN]JSON[CN]"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{self.dataset}_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump({
                'dataset': self.dataset,
                'timestamp': timestamp,
                'summaries': summaries
            }, f, indent=2)

        logger.info(f"\n[CN]: {filename}")

    def print_summary(self, summaries: List[Dict]):
        """print[CN]"""
        logger.info("\n" + "="*100)
        logger.info("[CN]")
        logger.info("="*100)
        logger.info(f"{'[CN]':<12} {'[CN]':<10} {'[CN](qps)':<15} {'Average latency(s)':<15} {'P95[CN](s)':<15} {'P99[CN](s)':<15}")
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
    parser = argparse.ArgumentParser(description='[CN]')
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET,
                       help='Dataset[CN]')
    parser.add_argument('--concurrent-levels', type=str, default='1,2,4,8,16',
                       help='[CN]（[CN]）')
    parser.add_argument('--queries-per-level', type=int, default=50,
                       help='[CN]Number of queries[CN]')

    args = parser.parse_args()

    # [CN]
    concurrent_levels = [int(x.strip()) for x in args.concurrent_levels.split(',')]

    # [CN]
    benchmark = ConcurrentBenchmark(dataset=args.dataset)
    benchmark.run_benchmark(concurrent_levels, args.queries_per_level)


if __name__ == "__main__":
    main()
