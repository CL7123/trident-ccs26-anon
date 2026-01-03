#!/usr/bin/env python3
"""
Fast standard HNSW index builder - Combines batch building efficiency with standard HNSW correctness
Generates truly hierarchically sparse index, but builds faster
"""

import numpy as np
import faiss
import struct
import time
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set, Tuple
import random
import argparse
import sys
from pathlib import Path

from domain_config import get_config, list_available_configs, SIFTSMALL, LAION, TRIPCLICK, MS_MARCO

class FastStandardHNSWBuilder:
    """Fast standard HNSW index builder"""

    def __init__(self, config=None, seed: int = 42):
        """
        Initialize HNSW parameters

        Args:
            config: DomainConfig object containing all configuration parameters
            seed: Random seed
        """
        if config is None:
            raise ValueError("Must provide config parameter")

        self.config = config
        self.M = config.M
        self.M0 = config.M * 2  # Maximum connections for layer 0
        self.efConstruction = config.efconstruction
        self.ml = 1.0 / np.log(2.0 * config.M)  # Layer assignment probability
        self.max_layers = config.layer
        self.seed = seed

        # graph structure
        self.graph = {layer: {} for layer in range(self.max_layers + 1)}
        self.node_levels = {}
        self.vectors = None
        self.entry_point = None

        # set random seed
        np.random.seed(seed)
        random.seed(seed)

        print(f"FastStandardHNSWBuilder initialized:")
        print(f"  Dataset: {config.vector_dimension}D, {config.num_docs:,} documents")
        print(f"  M = {self.M} (non-0 layer connections)")
        print(f"  M0 = {self.M0} (layer 0 connections)")
        print(f"  efConstruction = {self.efConstruction}")
        print(f"  efSearch = {config.efsearch}")
        print(f"  ml = {self.ml:.3f} (layer probability)")
        print(f"  Maximum layers = {self.max_layers}")
    
    def build_index(self, vectors: np.ndarray):
        """Fast build method assisted by FAISS"""
        self.vectors = vectors
        n_vectors = len(vectors)
        d = vectors.shape[1]

        print(f"\nStart fast HNSW index construction, {n_vectors} vectors total...")
        start_time = time.time()

        # 1. Assign levels (compliant with standard HNSW)
        print("1. Assigning node levels...")
        self._assign_levels(n_vectors)

        # 2. Build auxiliary FAISS index
        print("2. Building auxiliary FAISS index...")
        faiss_index = faiss.IndexHNSWFlat(d, self.M)
        faiss_index.hnsw.efConstruction = self.efConstruction
        faiss_index.add(vectors)

        # 3. Build hierarchically sparse graph structure
        print("3. Building hierarchically sparse graph structure...")
        self._build_sparse_graph(faiss_index)

        build_time = time.time() - start_time
        print(f"\nIndex construction complete, time elapsed: {build_time:.1f}s")

        # statistics information
        self._print_stats()
    
    def _assign_levels(self, n_vectors: int):
        """Assign node levels using standard HNSW probability distribution"""
        level_counts = defaultdict(int)

        for i in range(n_vectors):
            level = 0
            while level < self.max_layers and random.random() < self.ml:
                level += 1
            self.node_levels[i] = level

            for l in range(level + 1):
                level_counts[l] += 1

        # Select node from highest layer as entry point
        max_level_nodes = [n for n, l in self.node_levels.items() if l == self.max_layers]
        if max_level_nodes:
            self.entry_point = max_level_nodes[0]
        else:
            # If no nodes at highest layer, select from next highest
            for level in range(self.max_layers - 1, -1, -1):
                level_nodes = [n for n, l in self.node_levels.items() if l == level]
                if level_nodes:
                    self.entry_point = level_nodes[0]
                    break

        print(f"  Entry point: node_{self.entry_point} (level: {self.node_levels[self.entry_point]})")
        for level in sorted(level_counts.keys()):
            print(f"  Layer {level}: {level_counts[level]} nodes ({level_counts[level]/n_vectors*100:.1f}%)")
    
    def _build_sparse_graph(self, faiss_index: faiss.IndexHNSWFlat):
        """Build hierarchically sparse graph structure"""
        n_vectors = len(self.vectors)

        # set FAISS search parameters
        faiss_index.hnsw.efSearch = max(200, self.config.efsearch * 3)

        # build layer by layer
        for layer in range(self.max_layers + 1):
            print(f"\n  Building Layer {layer}...")

            # get all nodes in this layer
            nodes_in_layer = [i for i in range(n_vectors) if self.node_levels[i] >= layer]
            print(f"    Number of nodes: {len(nodes_in_layer)}")

            # batch search for neighbors
            batch_size = 1000
            for start_idx in range(0, len(nodes_in_layer), batch_size):
                end_idx = min(start_idx + batch_size, len(nodes_in_layer))
                batch_nodes = nodes_in_layer[start_idx:end_idx]
                batch_vectors = self.vectors[batch_nodes]

                # search for candidate neighbors
                k = self._get_search_k(layer)
                D, I = faiss_index.search(batch_vectors, k)

                # select neighbors for each node
                for i, node_id in enumerate(batch_nodes):
                    # filter out self and nodes not in this layer
                    valid_neighbors = []
                    for j, neighbor in enumerate(I[i]):
                        if neighbor >= 0 and neighbor != node_id and self.node_levels.get(neighbor, -1) >= layer:
                            valid_neighbors.append((D[i][j], neighbor))

                    # select neighbors
                    selected_neighbors = self._select_neighbors_standard(
                        node_id, valid_neighbors, layer
                    )

                    self.graph[layer][node_id] = selected_neighbors

                print(f"\r    Progress: {end_idx}/{len(nodes_in_layer)} ({end_idx/len(nodes_in_layer)*100:.1f}%)", end='')

            # statistics of average neighbors in this layer
            if nodes_in_layer:
                avg_neighbors = np.mean([len(self.graph[layer].get(n, [])) for n in nodes_in_layer])
                print(f"\n    Average neighbors: {avg_neighbors:.1f}")
    
    def _get_search_k(self, layer: int) -> int:
        """Return number of search candidates based on layer"""
        base_k = self.config.efconstruction
        if layer == 0:
            return min(base_k * 4, len(self.vectors))  # Layer 0 needs more candidates
        elif layer == 1:
            return min(base_k * 2, len(self.vectors))
        else:  # layer >= 2
            return min(base_k, len(self.vectors))
    
    def _select_neighbors_standard(self, node_id: int, candidates: List[Tuple[float, int]],
                                  layer: int) -> List[int]:
        """
        Standard HNSW neighbor selection strategy
        Higher layers select fewer, more diverse neighbors
        """
        if not candidates:
            return []

        # Set target neighbor count based on layer (fixed 3-layer structure: 0,1,2)
        if layer == 0:
            target_neighbors = min(self.M0, len(candidates))  # Layer 0: max connections (M*2)
        elif layer == 1:
            target_neighbors = min(self.M // 2, len(candidates))  # Layer 1: M/2 connections
        else:  # layer == 2
            target_neighbors = min(self.M // 4, len(candidates))  # Layer 2: M/4 connections

        # Sort by distance
        candidates.sort()

        # For higher layers, apply stricter diversity selection
        if layer >= 1:
            selected = []
            selected_set = set()

            for dist, neighbor in candidates:
                if len(selected) >= target_neighbors:
                    break

                # Check diversity
                should_add = True
                if layer >= 2 and len(selected) > 0:
                    # Higher layers require more diversity
                    min_diversity_dist = dist * 0.5  # At least 50% of distance apart
                    for s in selected[:5]:  # Check first few selected neighbors
                        if self._distance(self.vectors[neighbor], self.vectors[s]) < min_diversity_dist:
                            should_add = False
                            break

                if should_add and neighbor not in selected_set:
                    selected.append(neighbor)
                    selected_set.add(neighbor)

            # If too few selected, add some nearest neighbors
            if len(selected) < target_neighbors // 2:
                for _, neighbor in candidates:
                    if neighbor not in selected_set:
                        selected.append(neighbor)
                        selected_set.add(neighbor)
                        if len(selected) >= target_neighbors:
                            break

            return selected
        else:
            # Layer 0: simply select nearest neighbors
            return [neighbor for _, neighbor in candidates[:target_neighbors]]
    
    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate squared L2 distance"""
        return np.sum((a - b) ** 2)

    def _print_stats(self):
        """Print index statistics"""
        print("\nIndex statistics:")

        total_nodes = len(self.vectors)
        for layer in range(self.max_layers + 1):
            nodes_in_layer = [n for n, level in self.node_levels.items() if level >= layer]

            if nodes_in_layer:
                neighbor_counts = [len(self.graph[layer].get(n, [])) for n in nodes_in_layer]
                avg_neighbors = np.mean(neighbor_counts)
                min_neighbors = np.min(neighbor_counts)
                max_neighbors = np.max(neighbor_counts)

                print(f"  Layer {layer}: {len(nodes_in_layer)} nodes "
                      f"({len(nodes_in_layer)/total_nodes*100:.1f}%)")
                print(f"    Neighbors - avg: {avg_neighbors:.1f}, "
                      f"min: {min_neighbors}, max: {max_neighbors}")
            else:
                print(f"  Layer {layer}: 0 nodes")

        print(f"\n  Entry point: node_{self.entry_point} (highest layer: {self.node_levels[self.entry_point]})")
    
    def save_trident_format(self, output_dir: str, dataset_name: str = "dataset"):
        """Save in Trident format"""
        print(f"\nSaving Trident format to: {output_dir}")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        ntotal = len(self.vectors)
        d = self.vectors.shape[1]

        # Save node file
        node_file = f"{output_dir}/nodes.bin"
        with open(node_file, 'wb') as f:
            f.write(struct.pack('i', ntotal))
            f.write(struct.pack('i', d))
            f.write(struct.pack('i', self.entry_point))

            for i in range(ntotal):
                f.write(struct.pack('i', i))
                f.write(self.vectors[i].tobytes())

        print(f"  ✓ Node file: {node_file} ({Path(node_file).stat().st_size/1024/1024:.1f} MB)")

        # Save neighbor file
        neighbor_file = f"{output_dir}/neighbors.bin"
        num_levels = self.max_layers + 1
        maxM0 = self.M0  # Use M0 value from config

        with open(neighbor_file, 'wb') as f:
            f.write(struct.pack('i', ntotal))
            f.write(struct.pack('i', num_levels))
            f.write(struct.pack('i', maxM0))

            # Write data for each layer of each node
            for node_id in range(ntotal):
                for layer in range(num_levels):
                    f.write(struct.pack('i', node_id))
                    f.write(struct.pack('i', layer))

                    # Get neighbors of this node at this layer
                    neighbors = []
                    if node_id in self.node_levels and self.node_levels[node_id] >= layer:
                        neighbors = self.graph[layer].get(node_id, [])

                    # Pad to maxM0 length with -1
                    padded = list(neighbors) + [-1] * (maxM0 - len(neighbors))

                    for n in padded[:maxM0]:  # Ensure not exceeding maxM0
                        f.write(struct.pack('i', n))

        print(f"  ✓ Neighbor file: {neighbor_file} ({Path(neighbor_file).stat().st_size/1024/1024:.1f} MB)")

        return node_file, neighbor_file


def read_fvecs(filename):
    """Read fvecs format file"""
    fvecs = []
    with open(filename, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec = struct.unpack('f' * dim, f.read(4 * dim))
            fvecs.append(vec)
    return np.array(fvecs).astype('float32')


def main():
    """Main function - Fast standard HNSW index builder"""
    parser = argparse.ArgumentParser(description='Fast standard HNSW index builder')
    # Show available dataset configurations
    available_configs = list_available_configs()
    parser.add_argument('--dataset', type=str, required=True,
                       choices=available_configs,
                       help=f'Dataset name, options: {available_configs}')
    parser.add_argument('--data-path', type=str, help='Input data file path (fvecs format)')
    parser.add_argument('--output-dir', type=str, help='Output directory (default: ~/trident/dataset/dataset_name/)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')

    args = parser.parse_args()

    print("=== Fast Standard HNSW Index Builder ===")
    print("Combining batch building efficiency with standard HNSW hierarchical sparsity\n")

    # Get dataset configuration
    config = get_config(args.dataset)
    print(f"Using dataset configuration: {args.dataset}")
    print(f"  Vector dimension: {config.vector_dimension}")
    print(f"  Expected document count: {config.num_docs:,}")
    print(f"  HNSW parameters: M={config.M}, efConstruction={config.efconstruction}, layers={config.layer}")
    print()

    # Determine data path
    if args.data_path:
        base_path = args.data_path
    else:
        # Default path
        base_path = f"~/trident/dataset/{args.dataset}/base.fvecs"

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"~/trident/dataset/{args.dataset}"

    # Load data
    print(f"Loading data: {base_path}")
    vectors = read_fvecs(base_path)
    print(f"Data shape: {vectors.shape}")

    # Validate data dimension
    if vectors.shape[1] != config.vector_dimension:
        print(f"Warning: data dimension ({vectors.shape[1]}) does not match config dimension ({config.vector_dimension})")

    # Create builder
    builder = FastStandardHNSWBuilder(config=config, seed=args.seed)

    # Build index
    builder.build_index(vectors)

    # Save in Trident format
    builder.save_trident_format(output_dir, dataset_name=args.dataset)

    print("\nBuild complete!")
    print(f"Index files saved in: {output_dir}/")
    print(f"  - nodes.bin: node vector file")
    print(f"  - neighbors.bin: neighbor relationship file")


if __name__ == "__main__":
    main()