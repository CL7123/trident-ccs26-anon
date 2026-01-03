#!/usr/bin/env python3
"""
Concurrent performance testing script
Uses existing distributed-deploy/server.py (already supports concurrency)
Only requires client to send concurrent queries
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

# Add paths
sys.path.append('~/trident/distributed-deploy')
sys.path.append('~/trident/src')

# Import existing client
from client import DistributedClient
from config import SERVERS, DEFAULT_DATASET, CONCURRENT_LEVELS, QUERIES_PER_LEVEL

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ConcurrentBenchmark:
    """Concurrent performance testing"""

    def __init__(self, dataset: str = "siftsmall"):
        self.dataset = dataset
        # No longer using a single shared client, each query creates an independent connection
        self.servers_config = SERVERS
        self.results = defaultdict(list)

    def warmup(self, num_queries: int = 10):
        """Warmup: Send some queries to prepare the servers"""
        logger.info(f"Warmup: Sending {num_queries} queries...")

        for i in range(num_queries):
            node_id = random.randint(0, 9999)
            try:
                # Each query uses an independent client instance
                client = DistributedClient(dataset=self.dataset, servers_config=self.servers_config)
                if not client.connect_to_servers():
                    raise RuntimeError("Unable to connect to servers")
                client.test_distributed_query(node_id)
            except Exception as e:
                logger.warning(f"Warmup query {i+1} failed: {e}")

        logger.info("Warmup completed")

    def _calculate_concurrency_overlap(self, results: List[Dict]) -> float:
        """
        Calculate concurrency overlap: average number of queries executing simultaneously

        Algorithm:
        1. Collect time periods of all queries
        2. Count how many queries are executing at each time point
        3. Return the average

        Return value:
        - 1.0 = completely serial (no overlap)
        - N = average of N queries executing simultaneously
        """
        if not results:
            return 0.0

        # Collect all events (start and end)
        events = []
        for r in results:
            events.append(('start', r['start_time']))
            events.append(('end', r['end_time']))

        # Sort by time
        events.sort(key=lambda x: x[1])

        # Calculate concurrency for each time period
        current_concurrent = 0
        total_weighted_concurrent = 0
        total_time = 0
        last_time = events[0][1]

        for event_type, event_time in events:
            if event_time > last_time:
                # Calculate contribution of the previous time period
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
        """Execute a single query and return the result"""
        node_id = random.randint(0, 9999)
        query_id = f"benchmark_{time.time()}_{query_idx}"

        try:
            # Each query creates an independent client instance (independent connection)
            client = DistributedClient(dataset=self.dataset, servers_config=self.servers_config)
            if not client.connect_to_servers():
                raise RuntimeError("Unable to connect to servers")

            start_time = time.time()
            result = client.test_distributed_query(node_id)
            end_time = time.time()

            latency = end_time - start_time

            return {
                'success': result is not None,
                'latency': latency,
                'query_id': query_id,
                'node_id': node_id,
                'start_time': start_time,  # Record start time
                'end_time': end_time        # Record end time
            }
        except Exception as e:
            logger.error(f"Query {query_id} failed: {e}")
            return {
                'success': False,
                'latency': 0,
                'query_id': query_id,
                'error': str(e),
                'start_time': time.time(),
                'end_time': time.time()
            }

    def test_concurrent_level(self, concurrency: int, num_queries: int) -> Dict:
        """Test a specific concurrency level"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing concurrency level: {concurrency}")
        logger.info(f"Total queries: {num_queries}")
        logger.info(f"{'='*60}")

        results = []
        start_time = time.time()

        # Use ThreadPoolExecutor to execute queries concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            # Submit all queries
            futures = [executor.submit(self.query_single, i) for i in range(num_queries)]

            # Wait for completion and collect results
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                result = future.result()
                results.append(result)

                # Progress display
                if (i + 1) % 10 == 0 or (i + 1) == num_queries:
                    logger.info(f"Completed: {i+1}/{num_queries} queries")

        end_time = time.time()
        total_time = end_time - start_time

        # Aggregate results
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]

        if successful:
            latencies = [r['latency'] for r in successful]
            avg_latency = np.mean(latencies)
            p50_latency = np.percentile(latencies, 50)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            throughput = len(successful) / total_time

            # Calculate concurrency overlap (verify true concurrency)
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

        # Print results
        logger.info(f"\nResults summary:")
        logger.info(f"  Successful queries: {len(successful)}/{num_queries} ({summary['success_rate']:.1f}%)")
        logger.info(f"  Failed queries: {len(failed)}")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Throughput: {throughput:.2f} queries/sec")
        logger.info(f"  Avg latency: {avg_latency:.3f}s")
        logger.info(f"  P50 latency: {p50_latency:.3f}s")
        logger.info(f"  P95 latency: {p95_latency:.3f}s")
        logger.info(f"  P99 latency: {p99_latency:.3f}s")

        return summary

    def run_benchmark(self, concurrent_levels: List[int], queries_per_level: int):
        """Run complete benchmark test"""
        logger.info("="*80)
        logger.info("Concurrent Performance Benchmark")
        logger.info(f"Dataset: {self.dataset}")
        logger.info(f"Concurrency levels: {concurrent_levels}")
        logger.info(f"Queries per level: {queries_per_level}")
        logger.info("="*80)

        # Warmup
        self.warmup()

        # Test each concurrency level
        all_summaries = []
        for concurrency in concurrent_levels:
            summary = self.test_concurrent_level(concurrency, queries_per_level)
            all_summaries.append(summary)

            # Wait for system recovery
            logger.info(f"Waiting 5 seconds for system recovery...")
            time.sleep(5)

        # Save results
        self.save_results(all_summaries)

        # Print final summary
        self.print_summary(all_summaries)

        return all_summaries

    def save_results(self, summaries: List[Dict]):
        """Save test results to JSON file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{self.dataset}_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump({
                'dataset': self.dataset,
                'timestamp': timestamp,
                'summaries': summaries
            }, f, indent=2)

        logger.info(f"\nResults saved to: {filename}")

    def print_summary(self, summaries: List[Dict]):
        """Print performance summary table"""
        logger.info("\n" + "="*100)
        logger.info("Performance Summary")
        logger.info("="*100)
        logger.info(f"{'Concurrency':<12} {'Success Rate':<13} {'Throughput(qps)':<17} {'Avg Latency(s)':<17} {'P95 Latency(s)':<17} {'P99 Latency(s)':<17}")
        logger.info("-"*100)

        for s in summaries:
            logger.info(
                f"{s['concurrency']:<12} "
                f"{s['success_rate']:<13.1f} "
                f"{s['throughput']:<17.2f} "
                f"{s['avg_latency']:<17.3f} "
                f"{s['p95_latency']:<17.3f} "
                f"{s['p99_latency']:<17.3f}"
            )

        logger.info("="*100)


def main():
    parser = argparse.ArgumentParser(description='Concurrent performance benchmark')
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET,
                       help='Dataset name')
    parser.add_argument('--concurrent-levels', type=str, default='1,2,4,8,16',
                       help='Concurrency level list (comma-separated)')
    parser.add_argument('--queries-per-level', type=int, default=50,
                       help='Number of queries per concurrency level')

    args = parser.parse_args()

    # Parse concurrency levels
    concurrent_levels = [int(x.strip()) for x in args.concurrent_levels.split(',')]

    # Run test
    benchmark = ConcurrentBenchmark(dataset=args.dataset)
    benchmark.run_benchmark(concurrent_levels, args.queries_per_level)


if __name__ == "__main__":
    main()
