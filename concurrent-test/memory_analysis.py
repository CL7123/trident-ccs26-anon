#!/usr/bin/env python3
"""
Memory usage analysis script
Calculates Server and Client memory usage (based on data file sizes)
"""

import os
import sys
import json
import glob
import logging
from pathlib import Path
from typing import Dict, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [MemAnalysis] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class MemoryAnalyzer:
    """Memory usage analyzer"""

    def __init__(self, base_dir: str = "~/trident/dataset"):
        self.base_dir = base_dir
        self.datasets = ["siftsmall", "nfcorpus", "laion", "tripclick", "ms_marco"]
        self.results = {}

    def get_file_size_mb(self, filepath: str) -> float:
        """Get file size (MB)"""
        if os.path.exists(filepath):
            return os.path.getsize(filepath) / (1024 ** 2)
        return 0.0

    def get_file_size_gb(self, filepath: str) -> float:
        """Get file size (GB)"""
        if os.path.exists(filepath):
            return os.path.getsize(filepath) / (1024 ** 3)
        return 0.0

    def get_directory_size_gb(self, dirpath: str) -> float:
        """Get total size of all files in directory (GB)"""
        total_size = 0
        if os.path.exists(dirpath):
            for root, dirs, files in os.walk(dirpath):
                for file in files:
                    filepath = os.path.join(root, file)
                    try:
                        total_size += os.path.getsize(filepath)
                    except OSError as e:
                        logger.warning(f"Cannot access file {filepath}: {e}")
        return total_size / (1024 ** 3)

    def analyze_server_memory(self, dataset: str, server_id: int) -> Dict:
        """Analyze memory usage for a single server"""
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
            logger.warning(f"  nodes_shares.npy does not exist: {nodes_file}")

        # 2. neighbors_shares.npy
        neighbors_file = os.path.join(dataset_dir, "neighbors_shares.npy")
        if os.path.exists(neighbors_file):
            size_gb = self.get_file_size_gb(neighbors_file)
            server_data['neighbor_lists_gb'] = size_gb
            server_data['details'].append(f"neighbors_shares.npy: {size_gb:.3f} GB")
            logger.debug(f"  neighbors_shares.npy: {size_gb:.3f} GB")
        else:
            # Try old format neighbor_list_*.npy
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
                logger.warning(f"  neighbors_shares.npy does not exist: {neighbors_file}")

        # 3. triples (in separate triples directory)
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
            logger.warning(f"  triples directory does not exist: {triples_dir}")

        # Calculate total
        server_data['total_gb'] = (server_data['nodes_shares_gb'] +
                                   server_data['neighbor_lists_gb'] +
                                   server_data['triples_gb'])

        return server_data

    def analyze_client_memory(self, dataset: str) -> Dict:
        """Analyze client memory usage"""
        client_data = {
            'nodes_mb': 0.0,
            'dpf_keys_mb': 0.0,
            'total_mb': 0.0,
            'details': []
        }

        dataset_dir = os.path.join(self.base_dir, dataset)

        # 1. Original node vectors (for verification)
        # Try nodes.bin or nodes.npy
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
            logger.warning(f"  nodes file does not exist: {nodes_file_bin} or {nodes_file_npy}")

        # 2. DPF keys size (from previous test results)
        # Try to read client_cost test results
        cost_files = glob.glob(f"client_cost_{dataset}_*.json")
        if cost_files:
            latest_file = sorted(cost_files)[-1]
            try:
                with open(latest_file, 'r') as f:
                    data = json.load(f)
                    if 'statistics' in data and 'key_size' in data['statistics']:
                        # key_size is the size of a single key, needs × 3 (three servers)
                        single_key_kb = data['statistics']['key_size']['mean']
                        dpf_keys_mb = (single_key_kb * 3) / 1024  # KB → MB
                        client_data['dpf_keys_mb'] = dpf_keys_mb
                        client_data['details'].append(f"DPF keys (3×): {dpf_keys_mb:.2f} MB")
                        logger.debug(f"  DPF keys: {dpf_keys_mb:.2f} MB (from {latest_file})")
            except Exception as e:
                logger.warning(f"  Cannot read DPF key size: {e}")
        else:
            logger.warning(f"  client_cost test results for {dataset} not found, DPF key size set to 0")

        # Calculate total
        client_data['total_mb'] = client_data['nodes_mb'] + client_data['dpf_keys_mb']

        return client_data

    def analyze_dataset(self, dataset: str) -> Dict:
        """Analyze memory usage for a single dataset"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Analyzing dataset: {dataset.upper()}")
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
            logger.warning(f"Dataset directory does not exist: {dataset_dir}")
            return result

        result['exists'] = True

        # Analyze 3 servers
        logger.info("\nServer-side memory usage:")
        server_totals = []
        for server_id in [1, 2, 3]:
            logger.info(f"\n  Server {server_id}:")
            server_data = self.analyze_server_memory(dataset, server_id)
            result['servers'][server_id] = server_data
            server_totals.append(server_data['total_gb'])

            logger.info(f"    Node shares: {server_data['nodes_shares_gb']:.3f} GB")
            logger.info(f"    Neighbor lists: {server_data['neighbor_lists_gb']:.3f} GB")
            logger.info(f"    Triples: {server_data['triples_gb']:.3f} GB")
            logger.info(f"    Total: {server_data['total_gb']:.3f} GB")

        # Calculate average
        if server_totals:
            result['server_avg_gb'] = sum(server_totals) / len(server_totals)
            logger.info(f"\n  Server average memory: {result['server_avg_gb']:.3f} GB")

        # Analyze Client
        logger.info(f"\nClient-side memory usage:")
        client_data = self.analyze_client_memory(dataset)
        result['client'] = client_data

        logger.info(f"  Node vectors: {client_data['nodes_mb']:.2f} MB")
        logger.info(f"  DPF keys: {client_data['dpf_keys_mb']:.2f} MB")
        logger.info(f"  Total: {client_data['total_mb']:.2f} MB")

        return result

    def run_analysis(self):
        """Run complete memory analysis"""
        logger.info("="*80)
        logger.info("Starting memory usage analysis")
        logger.info("="*80)

        for dataset in self.datasets:
            result = self.analyze_dataset(dataset)
            if result['exists']:
                self.results[dataset] = result

        self.print_summary()
        self.save_results()

    def print_summary(self):
        """Print summary table"""
        logger.info(f"\n\n{'='*100}")
        logger.info("Memory Usage Summary")
        logger.info(f"{'='*100}\n")

        # Table header
        logger.info(f"{'Dataset':<15} {'Server (GB)':<15} {'Client (MB)':<15} {'Compass Server (GB)':<20} {'Compass Client (MB)':<20}")
        logger.info(f"{'-'*100}")

        # Compass data (for comparison)
        compass_data = {
            'laion': {'server': 0.95, 'client': 5.49},
            'siftsmall': {'server': 6.19, 'client': 35.84},
            'tripclick': {'server': 24.19, 'client': 77.48},
            'ms_marco': {'server': 193.50, 'client': 498.65}
        }

        for dataset, result in self.results.items():
            server_gb = result['server_avg_gb']
            client_mb = result['client']['total_mb']

            # Get Compass data
            compass = compass_data.get(dataset, {'server': '-', 'client': '-'})
            compass_server = compass['server'] if isinstance(compass['server'], str) else f"{compass['server']:.2f}"
            compass_client = compass['client'] if isinstance(compass['client'], str) else f"{compass['client']:.2f}"

            logger.info(f"{dataset.upper():<15} {server_gb:<15.3f} {client_mb:<15.2f} {compass_server:<20} {compass_client:<20}")

        logger.info(f"{'='*100}\n")

        # Markdown format
        logger.info("\nMarkdown table format:")
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
        """Save results to JSON file"""
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"memory_analysis_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Detailed results saved to: {filename}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Memory usage analysis')
    parser.add_argument('--base-dir', type=str, default='~/trident/dataset',
                       help='Dataset base directory')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed logs')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    analyzer = MemoryAnalyzer(base_dir=args.base_dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
