#!/usr/bin/env python3
"""
Communication cost analysis test script
Measures Client-Server communication overhead (size and time)
Calculates Server-Server MPC communication volume (theoretical value)
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

# Add paths
sys.path.append('~/trident/distributed-deploy')
sys.path.append('~/trident/src')

from client import DistributedClient
from config import SERVERS
from domain_config import get_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [CommCost] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class CommunicationCostBenchmark:
    """Communication cost testing"""

    def __init__(self, dataset: str = "siftsmall"):
        self.dataset = dataset
        self.config = get_config(dataset)
        self.client = DistributedClient(dataset=dataset, servers_config=SERVERS)
        self.results = []

        # Calculate MPC communication volume (theoretical value)
        self.mpc_size_mb = self._calculate_mpc_exchange_size()

        # Connect to servers
        if not self.client.connect_to_servers():
            raise RuntimeError("Unable to connect to servers")

        logger.info(f"Communication cost test initialized - Dataset: {dataset}")
        logger.info(f"MPC exchange data volume (theoretical): {self.mpc_size_mb:.2f} MB/query")

    def _calculate_mpc_exchange_size(self) -> float:
        """
        Calculate inter-server MPC communication volume (theoretical value)

        In Phase 3, each server exchanges e_shares and f_shares:
        - e_shares: int32[num_nodes]
        - f_shares: int32[num_nodes]
        - Each server sends to the other 2 servers
        - Total communication volume for 3 servers
        """
        num_nodes = self.config.num_docs

        # Size of each share (int32 = 4 bytes)
        share_size_bytes = num_nodes * 4

        # Each server sends: (e_shares + f_shares) × 2 target servers
        per_server_send_bytes = share_size_bytes * 2 * 2

        # Total communication volume for 3 servers
        total_bytes = per_server_send_bytes * 3

        # Convert to MB
        return total_bytes / (1024 ** 2)

    def measure_single_query(self, node_id: int) -> Dict:
        """Measure communication cost for a single query"""
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
            # 1. Generate DPF keys and measure size
            keys = self.client.dpf_wrapper.generate_keys('node', node_id)
            upload_size_bytes = sum(len(k) for k in keys)
            metrics['upload_size_mb'] = upload_size_bytes / (1024 ** 2)

            # 2. Generate query_id
            query_id = f'comm_benchmark_{time.time()}_{node_id}'

            # Start timing total time
            total_start = time.perf_counter()

            # 3. Measure upload time (send queries to all servers)
            upload_start = time.perf_counter()

            def query_server(server_id):
                request = {
                    'command': 'query_node_vector',
                    'dpf_key': keys[server_id - 1],
                    'query_id': query_id
                }
                # _send_request will send the data here
                response = self.client._send_request(server_id, request)
                return server_id, response

            # Send to all servers in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.client.connections)) as executor:
                futures = [executor.submit(query_server, sid) for sid in self.client.connections]

                # Upload time = time when all requests are sent
                # We wait for the first response to ensure all requests have been sent
                results = {}
                first_response_received = False

                for future in concurrent.futures.as_completed(futures):
                    try:
                        server_id, response = future.result()
                        results[server_id] = response

                        if not first_response_received:
                            # First response received, upload is complete
                            upload_time = time.perf_counter() - upload_start
                            metrics['upload_time_s'] = upload_time
                            first_response_received = True

                            # Start measuring download time
                            download_start = time.perf_counter()

                    except Exception as e:
                        logger.error(f"Error querying server: {e}")

            # 4. Download time = time of last response - time of first response
            download_time = time.perf_counter() - download_start
            metrics['download_time_s'] = download_time

            # 5. Calculate response size
            successful_responses = {sid: r for sid, r in results.items()
                                  if r and r.get('status') == 'success'}

            if len(successful_responses) < 2:
                logger.warning(f"Query {node_id} failed: fewer than 2 servers responded successfully")
                return metrics

            # Calculate download size (JSON serialized response)
            download_size_bytes = sum(len(json.dumps(r).encode())
                                     for r in successful_responses.values())
            metrics['download_size_mb'] = download_size_bytes / (1024 ** 2)

            # 6. Extract server-side timing information
            for server_id, result in successful_responses.items():
                timing = result.get('timing', {})
                metrics['phase1_time_s'] = timing.get('phase1_time', 0) / 1000
                metrics['phase2_time_s'] = timing.get('phase2_time', 0) / 1000
                metrics['phase3_time_s'] = timing.get('phase3_time', 0) / 1000
                metrics['phase4_time_s'] = timing.get('phase4_time', 0) / 1000
                break  # Only need timing from one server

            # 7. Calculate aggregate metrics
            total_time = time.perf_counter() - total_start
            metrics['total_time_s'] = total_time

            # Communication time = upload + download
            comm_time = metrics['upload_time_s'] + metrics['download_time_s']
            metrics['comm_time_s'] = comm_time

            # Computation time = phase1 + phase2 + phase4 (phase3 is communication)
            comp_time = (metrics['phase1_time_s'] +
                        metrics['phase2_time_s'] +
                        metrics['phase4_time_s'])
            metrics['comp_time_s'] = comp_time

            # Communication percentage
            if total_time > 0:
                metrics['comm_percentage'] = (comm_time / total_time) * 100

            metrics['success'] = True

        except Exception as e:
            logger.error(f"Error measuring query {node_id}: {e}")
            import traceback
            traceback.print_exc()
            metrics['success'] = False

        return metrics

    def run_benchmark(self, num_queries: int = 50):
        """Run benchmark test"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting communication cost test")
        logger.info(f"Dataset: {self.dataset}")
        logger.info(f"Number of queries: {num_queries}")
        logger.info(f"Number of documents: {self.config.num_docs:,}")
        logger.info(f"MPC exchange data volume (theoretical): {self.mpc_size_mb:.2f} MB/query")
        logger.info(f"{'='*80}\n")

        # Warmup
        logger.info("Warmup queries...")
        for i in range(5):
            node_id = random.randint(0, min(9999, self.config.num_docs - 1))
            try:
                self.measure_single_query(node_id)
            except Exception as e:
                logger.warning(f"Warmup query {i+1} failed: {e}")

        logger.info("Warmup completed, starting formal test\n")

        # Formal test
        for i in range(num_queries):
            node_id = random.randint(0, min(9999, self.config.num_docs - 1))

            logger.info(f"Test query {i+1}/{num_queries} (node_id={node_id})...")
            metrics = self.measure_single_query(node_id)

            if metrics['success']:
                self.results.append(metrics)
                logger.info(f"  ✓ Upload: {metrics['upload_size_mb']:.3f}MB/{metrics['upload_time_s']:.3f}s, "
                          f"Download: {metrics['download_size_mb']:.3f}MB/{metrics['download_time_s']:.3f}s, "
                          f"Comm%: {metrics['comm_percentage']:.1f}%")
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

        logger.info(f"\n{'='*100}")
        logger.info("Communication Cost Analysis - Statistical Summary")
        logger.info(f"{'='*100}")
        logger.info(f"Dataset: {self.dataset}")
        logger.info(f"Successful queries: {len(self.results)}")
        logger.info(f"{'='*100}\n")

        # Calculate statistics
        def calc_stats(values):
            """Calculate mean and standard deviation, remove outliers"""
            arr = np.array(values)
            mean = np.mean(arr)
            std = np.std(arr)
            # Remove outliers exceeding 3 standard deviations
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
        upload_sizes = [r['upload_size_mb'] for r in self.results]
        download_sizes = [r['download_size_mb'] for r in self.results]
        upload_times = [r['upload_time_s'] for r in self.results]
        download_times = [r['download_time_s'] for r in self.results]
        comm_times = [r['comm_time_s'] for r in self.results]
        comp_times = [r['comp_time_s'] for r in self.results]
        total_times = [r['total_time_s'] for r in self.results]
        comm_percentages = [r['comm_percentage'] for r in self.results]

        # Compute statistics
        upload_size_stats = calc_stats(upload_sizes)
        download_size_stats = calc_stats(download_sizes)
        upload_time_stats = calc_stats(upload_times)
        download_time_stats = calc_stats(download_times)
        comm_time_stats = calc_stats(comm_times)
        comp_time_stats = calc_stats(comp_times)
        total_time_stats = calc_stats(total_times)
        comm_pct_stats = calc_stats(comm_percentages)

        # MPC size is a fixed value
        mpc_size = self.mpc_size_mb

        # Print results table
        logger.info(f"{'Metric':<30} {'Mean':<15} {'Std Dev':<15} {'Min':<15} {'Max':<15}")
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

        # Print paper table format
        logger.info("Paper table format:")
        logger.info(f"| {self.dataset:<10} | "
                   f"{upload_size_stats['mean']:>5.2f} ± {upload_size_stats['std']:>4.2f} | "
                   f"{download_size_stats['mean']:>5.2f} ± {download_size_stats['std']:>4.2f} | "
                   f"{comm_time_stats['mean']:>5.2f} ± {comm_time_stats['std']:>4.2f} | "
                   f"{mpc_size:>8.2f} | "
                   f"{comp_time_stats['mean']:>5.2f} ± {comp_time_stats['std']:>4.2f} | "
                   f"{total_time_stats['mean']:>5.1f} ± {total_time_stats['std']:>4.1f} | "
                   f"{comm_pct_stats['mean']:>5.1f}% |")

        # Store statistics
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
        """Save test results"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save detailed results
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

        logger.info(f"Detailed results saved to: {detail_filename}")

    def cleanup(self):
        """Clean up resources"""
        self.client.disconnect_from_servers()


def main():
    parser = argparse.ArgumentParser(description='Communication cost analysis test')
    parser.add_argument('--dataset', type=str, default='siftsmall',
                       help='Dataset name (siftsmall, nfcorpus, laion, tripclick)')
    parser.add_argument('--num-queries', type=int, default=50,
                       help='Number of test queries')

    args = parser.parse_args()

    try:
        benchmark = CommunicationCostBenchmark(dataset=args.dataset)
        benchmark.run_benchmark(num_queries=args.num_queries)
        benchmark.cleanup()
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
