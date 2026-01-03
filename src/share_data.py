#!/usr/bin/env python3
"""
Secret sharing for LAION dataset
- Node vectors: 100k × 512 dimensions
- Neighbor lists: linearized storage (node_id * 3 + layer)
"""

import sys
import os
import struct
import numpy as np
import json
import time
import argparse
from typing import Dict, Tuple

# Import domain configuration
from domain_config import get_config as get_domain_config, list_available_configs, SIFTSMALL, LAION, TRIPCLICK, MS_MARCO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add parent directory to path for imports
from basic_functionalities import MPC23SSS, get_config as get_mpc_config


class DatasetLoader:
    """Load and parse dataset"""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_nodes(self) -> np.ndarray:
        """Load node vector data"""
        print("Loading node vectors...")
        nodes_path = os.path.join(self.data_dir, "nodes.bin")

        with open(nodes_path, 'rb') as f:
            # Read header
            num_nodes = struct.unpack('<I', f.read(4))[0]
            vector_dim = struct.unpack('<I', f.read(4))[0]

            print(f"  Nodes: {num_nodes:,}")
            print(f"  Vector dimension: {vector_dim}")

            # Skip extra header bytes
            f.read(4)  # Usually 12 or other metadata

            # Read all vectors
            total_floats = num_nodes * vector_dim
            vectors = struct.unpack(f'<{total_floats}f', f.read(total_floats * 4))

            # Reshape to 2D array
            nodes = np.array(vectors, dtype=np.float32).reshape(num_nodes, vector_dim)

        return nodes
    
    def load_neighbors(self) -> Tuple[np.ndarray, int, int]:
        """Load neighbor list data"""
        print("\nLoading neighbor lists...")
        neighbors_path = os.path.join(self.data_dir, "neighbors.bin")

        with open(neighbors_path, 'rb') as f:
            # Read header
            num_nodes = struct.unpack('<I', f.read(4))[0]
            num_layers = struct.unpack('<I', f.read(4))[0]
            max_neighbors = struct.unpack('<I', f.read(4))[0]
            _ = struct.unpack('<I', f.read(4))[0]  # Skip extra 0

            print(f"  Nodes: {num_nodes:,}")
            print(f"  Layers: {num_layers}")
            print(f"  Max neighbors: {max_neighbors}")

            # Data format: each node has (3 layers × 128 neighbors + 2 extra values) integers
            ints_per_node = num_layers * max_neighbors + 2

            # Read all data
            total_ints = num_nodes * ints_per_node
            print(f"  Reading {total_ints:,} integers...")
            all_data = struct.unpack(f'<{total_ints}I', f.read(total_ints * 4))

            # Parse to structured data
            neighbors = []

            for node_id in range(num_nodes):
                # Calculate starting position of this node's data
                start_idx = node_id * ints_per_node

                node_neighbors = []
                for layer in range(num_layers):
                    # Get 128 neighbors for this layer
                    layer_start = start_idx + layer * max_neighbors
                    layer_data = all_data[layer_start:layer_start + max_neighbors]

                    # Filter out padding values (-1 represented as 4294967295)
                    actual_neighbors = [x for x in layer_data if x != 4294967295 and x < num_nodes]
                    node_neighbors.append(actual_neighbors)

                neighbors.append(node_neighbors)

                # Progress display
                if (node_id + 1) % 10000 == 0:
                    print(f"  Processed {node_id + 1:,} / {num_nodes:,} nodes")

            return neighbors, num_layers, max_neighbors


class DatasetSecretSharing:
    """Secret sharing for dataset"""

    def __init__(self, dataset_name: str):
        # Get domain configuration
        self.domain_config = get_domain_config(dataset_name)

        # Create MPC configuration using domain configuration
        self.mpc_config = get_mpc_config(dataset_name)
        self.mpc = MPC23SSS(self.mpc_config)

        # Use parameters from domain configuration
        self.field_size = self.domain_config.prime
        # Dynamically set scale factor based on output bits
        # Use different scaling strategies for different datasets
        if dataset_name == "siftsmall":
            # siftsmall data range [0, 180], use smaller scale factor
            self.scale_factor = 2 ** 20  # Sufficient precision for storing 180
        elif dataset_name == "tripclick":
            # tripclick data range approx [-5, 5], use smaller scale factor to avoid overflow
            self.scale_factor = 2 ** 20  # 1M scale factor, reduced to avoid overflow in 768-dim vector calculations
        elif dataset_name == "nfcorpus":
            # nfcorpus data range approx [-12, 4], use smaller scale factor
            self.scale_factor = 2 ** 22  # 4M scale factor, avoid overflow
        else:
            # Other datasets (e.g., LAION) data range [-1, 1], use original scale factor
            self.scale_factor = 2 ** (self.domain_config.output_bits - 2)

    def float_to_field(self, value: float) -> int:
        """Convert floating-point number to finite field"""
        # Scale and convert to integer, then modulo field_size
        # Python's modulo operation automatically handles negative numbers, mapping them to [0, field_size) range
        scaled = int(value * self.scale_factor)
        return scaled % self.field_size
    
    def share_nodes(self, nodes: np.ndarray) -> Dict[int, np.ndarray]:
        """Secret sharing for node vectors"""
        num_nodes, vector_dim = nodes.shape
        print(f"\nStarting node vector secret sharing...")
        print(f"  Data shape: {nodes.shape}")

        # Initialize server share arrays
        server_shares = {
            1: np.zeros((num_nodes, vector_dim), dtype=np.uint32),
            2: np.zeros((num_nodes, vector_dim), dtype=np.uint32),
            3: np.zeros((num_nodes, vector_dim), dtype=np.uint32)
        }

        # Progress display
        start_time = time.time()

        for i in range(num_nodes):
            # Secret share each dimension of each vector
            for j in range(vector_dim):
                # Convert to finite field
                field_value = self.float_to_field(nodes[i, j])

                # Generate secret shares
                shares = self.mpc.share_secret(field_value)

                # Distribute to servers
                for server_id in range(1, 4):
                    server_shares[server_id][i, j] = shares[server_id-1].value

            # Progress display
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (num_nodes - i - 1) / rate
                print(f"  Progress: {i+1:,}/{num_nodes:,} ({(i+1)/num_nodes*100:.1f}%) - "
                      f"Rate: {rate:.0f} nodes/s - Remaining: {eta:.0f}s")

        return server_shares

    def share_neighbors_linearized(self, neighbors: list, num_layers: int,
                                   max_neighbors: int) -> Dict[int, np.ndarray]:
        """Linearize and secret share neighbor lists"""
        num_nodes = len(neighbors)
        total_entries = num_nodes * num_layers

        print(f"\nStarting neighbor list secret sharing...")
        print(f"  Linearized array size: {total_entries:,} × {max_neighbors}")

        # Initialize server share arrays
        server_shares = {
            1: np.zeros((total_entries, max_neighbors), dtype=np.uint32),
            2: np.zeros((total_entries, max_neighbors), dtype=np.uint32),
            3: np.zeros((total_entries, max_neighbors), dtype=np.uint32)
        }

        # Progress display
        start_time = time.time()
        processed = 0

        for node_id in range(num_nodes):
            for layer in range(num_layers):
                # Calculate linear index
                linear_idx = node_id * num_layers + layer

                # Get neighbor list at this position
                neighbor_list = neighbors[node_id][layer] if layer < len(neighbors[node_id]) else []

                # Pad to max_neighbors, use -1 as padding value
                padding_value = self.field_size - 1  # -1 represented in finite field
                padded_neighbors = neighbor_list + [padding_value] * (max_neighbors - len(neighbor_list))

                # Secret share each neighbor ID
                for j in range(max_neighbors):
                    neighbor_id = padded_neighbors[j]

                    # Generate secret shares
                    shares = self.mpc.share_secret(neighbor_id)

                    # Distribute to servers
                    for server_id in range(1, 4):
                        server_shares[server_id][linear_idx, j] = shares[server_id-1].value

                processed += 1

                # Progress display
                if processed % 10000 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed
                    eta = (total_entries - processed) / rate
                    print(f"  Progress: {processed:,}/{total_entries:,} ({processed/total_entries*100:.1f}%) - "
                          f"Rate: {rate:.0f} entries/s - Remaining: {eta:.0f}s")

        return server_shares


def main():
    """Main function: load data and perform secret sharing"""
    # Set command line parameters
    parser = argparse.ArgumentParser(description='Secret sharing for dataset')
    available_configs = list_available_configs()
    parser.add_argument('--dataset', type=str, default=SIFTSMALL,
                        choices=available_configs,
                        help=f'Dataset name, options: {available_configs}')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Dataset directory path')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory path')
    args = parser.parse_args()

    print(f"=== Dataset Secret Sharing: {args.dataset} ===\n")

    # Get domain configuration
    domain_config = get_domain_config(args.dataset)
    print(f"Using dataset configuration: {args.dataset}")
    print(f"  Vector dimension: {domain_config.vector_dimension}")
    print(f"  Expected documents: {domain_config.num_docs:,}")
    print(f"  Prime field: p = {domain_config.prime:,}")
    print(f"  Output bits: {domain_config.output_bits}")
    print()

    # Set default paths
    # Use relative path from project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.data_dir is None:
        data_dir = os.path.join(project_root, "datasets", args.dataset)
    else:
        data_dir = args.data_dir

    if args.output_dir is None:
        output_dir = os.path.join(project_root, "datasets", args.dataset)
    else:
        output_dir = args.output_dir

    # 1. Load data
    loader = DatasetLoader(data_dir)

    # Load node vectors
    nodes = loader.load_nodes()

    # Load neighbor lists
    neighbors, num_layers, max_neighbors = loader.load_neighbors()

    # 2. Perform secret sharing
    sharer = DatasetSecretSharing(args.dataset)

    # Share node vectors
    node_shares = sharer.share_nodes(nodes)

    # Share neighbor lists
    neighbor_shares = sharer.share_neighbors_linearized(neighbors, num_layers, max_neighbors)

    # 3. Save to files

    print("\nSaving secret shares...")
    for server_id in range(1, 4):
        server_dir = os.path.join(output_dir, f"server_{server_id}")
        os.makedirs(server_dir, exist_ok=True)

        # Save node shares
        nodes_path = os.path.join(server_dir, "nodes_shares.npy")
        np.save(nodes_path, node_shares[server_id])
        print(f"  Server {server_id} node shares: {nodes_path}")

        # Save neighbor shares
        neighbors_path = os.path.join(server_dir, "neighbors_shares.npy")
        np.save(neighbors_path, neighbor_shares[server_id])
        print(f"  Server {server_id} neighbor shares: {neighbors_path}")

        # Save metadata
        metadata = {
            "dataset": args.dataset,
            "num_nodes": len(nodes),
            "num_layers": num_layers,
            "max_neighbors": max_neighbors,
            "vector_dim": nodes.shape[1],
            "indexing": "linear",
            "index_formula": "node_id * num_layers + layer",
            "scale_factor": sharer.scale_factor,
            "field_size": sharer.field_size,
            "domain_bits": domain_config.domain_bits,
            "domain_size": domain_config.domain_size,
            "output_bits": domain_config.output_bits,
            "M": domain_config.M,
            "efconstruction": domain_config.efconstruction,
            "efsearch": domain_config.efsearch,
            "total_neighbor_entries": len(nodes) * num_layers,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        metadata_path = os.path.join(server_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Server {server_id} metadata: {metadata_path}")

    print(f"\n✅ {args.dataset} dataset secret sharing complete!")
    print(f"\nOutput directory: {output_dir}")
    print("Usage examples:")
    print("  python share_data.py --dataset siftsmall")
    print("  python share_data.py --dataset laion")
    print("  python share_data.py --dataset tripclick --output-dir /custom/path")


if __name__ == "__main__":
    main()