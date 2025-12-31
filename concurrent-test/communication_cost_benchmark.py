#!/usr/bin/env python3
"""
[CN]
[CN] Client-Server [CN]（[CN]）
calculate Server-Server MPC [CN]（[CN]）
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

# [CN]
sys.path.append('~/trident/distributed-deploy')
sys.path.append('~/trident/src')

from client import DistributedClient
from config import SERVERS
from domain_config import get_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [CommCost] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class CommunicationCostBenchmark:
    """[CN]"""

    def __init__(self, dataset: str = "siftsmall"):
        self.dataset = dataset
        self.config = get_config(dataset)
        self.client = DistributedClient(dataset=dataset, servers_config=SERVERS)
        self.results = []

        # calculateMPC[CN]（[CN]）
        self.mpc_size_mb = self._calculate_mpc_exchange_size()

        # connect[CN]
        if not self.client.connect_to_servers():
            raise RuntimeError("[CN]connect[CN]")

        logger.info(f"[CN]initialize[CN] - Dataset: {dataset}")
        logger.info(f"MPC[CN]（[CN]）: {self.mpc_size_mb:.2f} MB/query")

    def _calculate_mpc_exchange_size(self) -> float:
        """
        calculateServer[CN]MPC[CN]（[CN]）

        Phase 3 [CN]server[CN] e_shares [CN] f_shares:
        - e_shares: int32[num_nodes]
        - f_shares: int32[num_nodes]
        - [CN]serversend[CN]2[CN]servers
        - 3[CN]servers[CN]
        """
        num_nodes = self.config.num_docs

        # [CN]share[CN]（int32 = 4 bytes）
        share_size_bytes = num_nodes * 4

        # [CN]serversend: (e_shares + f_shares) × 2[CN]servers
        per_server_send_bytes = share_size_bytes * 2 * 2

        # 3[CN]servers[CN]
        total_bytes = per_server_send_bytes * 3

        # [CN]MB
        return total_bytes / (1024 ** 2)

    def measure_single_query(self, node_id: int) -> Dict:
        """[CN]"""
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
            # 1. [CN]DPF keys[CN]
            keys = self.client.dpf_wrapper.generate_keys('node', node_id)
            upload_size_bytes = sum(len(k) for k in keys)
            metrics['upload_size_mb'] = upload_size_bytes / (1024 ** 2)

            # 2. [CN]query_id
            query_id = f'comm_benchmark_{time.time()}_{node_id}'

            # [CN]
            total_start = time.perf_counter()

            # 3. [CN]（sendqueries[CN]servers）
            upload_start = time.perf_counter()

            def query_server(server_id):
                request = {
                    'command': 'query_node_vector',
                    'dpf_key': keys[server_id - 1],
                    'query_id': query_id
                }
                # [CN]_send_request[CN]send[CN]
                response = self.client._send_request(server_id, request)
                return server_id, response

            # [CN]send[CN]servers
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.client.connections)) as executor:
                futures = [executor.submit(query_server, sid) for sid in self.client.connections]

                # [CN] = [CN]send[CN]
                # [CN]response[CN]send
                results = {}
                first_response_received = False

                for future in concurrent.futures.as_completed(futures):
                    try:
                        server_id, response = future.result()
                        results[server_id] = response

                        if not first_response_received:
                            # [CN]response[CN]，[CN]
                            upload_time = time.perf_counter() - upload_start
                            metrics['upload_time_s'] = upload_time
                            first_response_received = True

                            # [CN]
                            download_start = time.perf_counter()

                    except Exception as e:
                        logger.error(f"[CN]: {e}")

            # 4. [CN] = [CN]response[CN] - [CN]response[CN]
            download_time = time.perf_counter() - download_start
            metrics['download_time_s'] = download_time

            # 5. calculateresponse[CN]
            successful_responses = {sid: r for sid, r in results.items()
                                  if r and r.get('status') == 'success'}

            if len(successful_responses) < 2:
                logger.warning(f"[CN] {node_id} [CN]：[CN]2[CN]")
                return metrics

            # calculatedownload[CN]（JSON[CN]response）
            download_size_bytes = sum(len(json.dumps(r).encode())
                                     for r in successful_responses.values())
            metrics['download_size_mb'] = download_size_bytes / (1024 ** 2)

            # 6. [CN]server[CN]timing[CN]
            for server_id, result in successful_responses.items():
                timing = result.get('timing', {})
                metrics['phase1_time_s'] = timing.get('phase1_time', 0) / 1000
                metrics['phase2_time_s'] = timing.get('phase2_time', 0) / 1000
                metrics['phase3_time_s'] = timing.get('phase3_time', 0) / 1000
                metrics['phase4_time_s'] = timing.get('phase4_time', 0) / 1000
                break  # [CN]server[CN]timing

            # 7. calculate[CN]
            total_time = time.perf_counter() - total_start
            metrics['total_time_s'] = total_time

            # [CN] = upload + download
            comm_time = metrics['upload_time_s'] + metrics['download_time_s']
            metrics['comm_time_s'] = comm_time

            # calculate[CN] = phase1 + phase2 + phase4 (phase3[CN])
            comp_time = (metrics['phase1_time_s'] +
                        metrics['phase2_time_s'] +
                        metrics['phase4_time_s'])
            metrics['comp_time_s'] = comp_time

            # [CN]
            if total_time > 0:
                metrics['comm_percentage'] = (comm_time / total_time) * 100

            metrics['success'] = True

        except Exception as e:
            logger.error(f"[CN] {node_id} [CN]: {e}")
            import traceback
            traceback.print_exc()
            metrics['success'] = False

        return metrics

    def run_benchmark(self, num_queries: int = 50):
        """[CN]"""
        logger.info(f"\n{'='*80}")
        logger.info(f"[CN]")
        logger.info(f"Dataset: {self.dataset}")
        logger.info(f"Number of queries[CN]: {num_queries}")
        logger.info(f"Number of documents[CN]: {self.config.num_docs:,}")
        logger.info(f"MPC[CN]（[CN]）: {self.mpc_size_mb:.2f} MB/query")
        logger.info(f"{'='*80}\n")

        # [CN]
        logger.info("[CN]...")
        for i in range(5):
            node_id = random.randint(0, min(9999, self.config.num_docs - 1))
            try:
                self.measure_single_query(node_id)
            except Exception as e:
                logger.warning(f"[CN] {i+1} [CN]: {e}")

        logger.info("[CN]，[CN]\n")

        # [CN]
        for i in range(num_queries):
            node_id = random.randint(0, min(9999, self.config.num_docs - 1))

            logger.info(f"[CN] {i+1}/{num_queries} (node_id={node_id})...")
            metrics = self.measure_single_query(node_id)

            if metrics['success']:
                self.results.append(metrics)
                logger.info(f"  ✓ Upload: {metrics['upload_size_mb']:.3f}MB/{metrics['upload_time_s']:.3f}s, "
                          f"Download: {metrics['download_size_mb']:.3f}MB/{metrics['download_time_s']:.3f}s, "
                          f"Comm%: {metrics['comm_percentage']:.1f}%")
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

        logger.info(f"\n{'='*100}")
        logger.info("[CN] - [CN]")
        logger.info(f"{'='*100}")
        logger.info(f"Dataset: {self.dataset}")
        logger.info(f"[CN]Number of queries: {len(self.results)}")
        logger.info(f"{'='*100}\n")

        # calculate[CN]
        def calc_stats(values):
            """calculate[CN]，[CN]"""
            arr = np.array(values)
            mean = np.mean(arr)
            std = np.std(arr)
            # [CN]3[CN]
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
        upload_sizes = [r['upload_size_mb'] for r in self.results]
        download_sizes = [r['download_size_mb'] for r in self.results]
        upload_times = [r['upload_time_s'] for r in self.results]
        download_times = [r['download_time_s'] for r in self.results]
        comm_times = [r['comm_time_s'] for r in self.results]
        comp_times = [r['comp_time_s'] for r in self.results]
        total_times = [r['total_time_s'] for r in self.results]
        comm_percentages = [r['comm_percentage'] for r in self.results]

        # calculate[CN]
        upload_size_stats = calc_stats(upload_sizes)
        download_size_stats = calc_stats(download_sizes)
        upload_time_stats = calc_stats(upload_times)
        download_time_stats = calc_stats(download_times)
        comm_time_stats = calc_stats(comm_times)
        comp_time_stats = calc_stats(comp_times)
        total_time_stats = calc_stats(total_times)
        comm_pct_stats = calc_stats(comm_percentages)

        # MPC size[CN]
        mpc_size = self.mpc_size_mb

        # print[CN]
        logger.info(f"{'[CN]':<30} {'[CN]':<15} {'[CN]':<15} {'[CN]':<15} {'[CN]':<15}")
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

        # print[CN]
        logger.info("[CN]:")
        logger.info(f"| {self.dataset:<10} | "
                   f"{upload_size_stats['mean']:>5.2f} ± {upload_size_stats['std']:>4.2f} | "
                   f"{download_size_stats['mean']:>5.2f} ± {download_size_stats['std']:>4.2f} | "
                   f"{comm_time_stats['mean']:>5.2f} ± {comm_time_stats['std']:>4.2f} | "
                   f"{mpc_size:>8.2f} | "
                   f"{comp_time_stats['mean']:>5.2f} ± {comp_time_stats['std']:>4.2f} | "
                   f"{total_time_stats['mean']:>5.1f} ± {total_time_stats['std']:>4.1f} | "
                   f"{comm_pct_stats['mean']:>5.1f}% |")

        # [CN]
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
        """[CN]Test results"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # [CN]
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
        benchmark = CommunicationCostBenchmark(dataset=args.dataset)
        benchmark.run_benchmark(num_queries=args.num_queries)
        benchmark.cleanup()
    except Exception as e:
        logger.error(f"[CN]: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
