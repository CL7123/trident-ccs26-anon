#!/usr/bin/env python3
"""
[CN]HNSW[CN] - [CN]HNSW[CN]
[CN]，[CN]
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
    """[CN]HNSW[CN]"""
    
    def __init__(self, config=None, seed: int = 42):
        """
        initializeHNSW[CN]
        
        Args:
            config: DomainConfig[CN]，[CN]
            seed: [CN]
        """
        if config is None:
            raise ValueError("[CN]config[CN]")
            
        self.config = config
        self.M = config.M
        self.M0 = config.M * 2  # Layer 0[CN]connect[CN]
        self.efConstruction = config.efconstruction
        self.ml = 1.0 / np.log(2.0 * config.M)  # [CN]allocate[CN]
        self.max_layers = config.layer
        self.seed = seed
        
        # [CN]
        self.graph = {layer: {} for layer in range(self.max_layers + 1)}
        self.node_levels = {}
        self.vectors = None
        self.entry_point = None
        
        # [CN]
        np.random.seed(seed)
        random.seed(seed)
        
        print(f"FastStandardHNSWBuilderinitialize:")
        print(f"  Dataset: {config.vector_dimension}[CN], {config.num_docs:,}[CN]")
        print(f"  M = {self.M} ([CN]0[CN]connect[CN])")
        print(f"  M0 = {self.M0} ([CN]0[CN]connect[CN])")
        print(f"  efConstruction = {self.efConstruction}")
        print(f"  efSearch = {config.efsearch}")
        print(f"  ml = {self.ml:.3f} ([CN])")
        print(f"  [CN] = {self.max_layers}")
    
    def build_index(self, vectors: np.ndarray):
        """[CN]FAISS[CN]"""
        self.vectors = vectors
        n_vectors = len(vectors)
        d = vectors.shape[1]
        
        print(f"\n[CN]HNSW[CN]，[CN]{n_vectors}[CN]...")
        start_time = time.time()
        
        # 1. allocate[CN]（[CN]HNSW）
        print("1. allocate[CN]...")
        self._assign_levels(n_vectors)
        
        # 2. [CN]FAISS[CN]
        print("2. [CN]FAISS[CN]...")
        faiss_index = faiss.IndexHNSWFlat(d, self.M)
        faiss_index.hnsw.efConstruction = self.efConstruction
        faiss_index.add(vectors)
        
        # 3. [CN]
        print("3. [CN]...")
        self._build_sparse_graph(faiss_index)
        
        build_time = time.time() - start_time
        print(f"\n[CN]，[CN]: {build_time:.1f}[CN]")
        
        # [CN]
        self._print_stats()
    
    def _assign_levels(self, n_vectors: int):
        """allocate[CN]，[CN]HNSW[CN]"""
        level_counts = defaultdict(int)
        
        for i in range(n_vectors):
            level = 0
            while level < self.max_layers and random.random() < self.ml:
                level += 1
            self.node_levels[i] = level
            
            for l in range(level + 1):
                level_counts[l] += 1
        
        # [CN]
        max_level_nodes = [n for n, l in self.node_levels.items() if l == self.max_layers]
        if max_level_nodes:
            self.entry_point = max_level_nodes[0]
        else:
            # [CN]，[CN]
            for level in range(self.max_layers - 1, -1, -1):
                level_nodes = [n for n, l in self.node_levels.items() if l == level]
                if level_nodes:
                    self.entry_point = level_nodes[0]
                    break
        
        print(f"  [CN]: node_{self.entry_point} ([CN]: {self.node_levels[self.entry_point]})")
        for level in sorted(level_counts.keys()):
            print(f"  Layer {level}: {level_counts[level]} [CN] ({level_counts[level]/n_vectors*100:.1f}%)")
    
    def _build_sparse_graph(self, faiss_index: faiss.IndexHNSWFlat):
        """[CN]"""
        n_vectors = len(self.vectors)
        
        # [CN]FAISS[CN]
        faiss_index.hnsw.efSearch = max(200, self.config.efsearch * 3)
        
        # [CN]
        for layer in range(self.max_layers + 1):
            print(f"\n  [CN]Layer {layer}...")
            
            # [CN]
            nodes_in_layer = [i for i in range(n_vectors) if self.node_levels[i] >= layer]
            print(f"    [CN]: {len(nodes_in_layer)}")
            
            # [CN]
            batch_size = 1000
            for start_idx in range(0, len(nodes_in_layer), batch_size):
                end_idx = min(start_idx + batch_size, len(nodes_in_layer))
                batch_nodes = nodes_in_layer[start_idx:end_idx]
                batch_vectors = self.vectors[batch_nodes]
                
                # [CN]
                k = self._get_search_k(layer)
                D, I = faiss_index.search(batch_vectors, k)
                
                # [CN]
                for i, node_id in enumerate(batch_nodes):
                    # [CN]
                    valid_neighbors = []
                    for j, neighbor in enumerate(I[i]):
                        if neighbor >= 0 and neighbor != node_id and self.node_levels.get(neighbor, -1) >= layer:
                            valid_neighbors.append((D[i][j], neighbor))
                    
                    # [CN]
                    selected_neighbors = self._select_neighbors_standard(
                        node_id, valid_neighbors, layer
                    )
                    
                    self.graph[layer][node_id] = selected_neighbors
                
                print(f"\r    [CN]: {end_idx}/{len(nodes_in_layer)} ({end_idx/len(nodes_in_layer)*100:.1f}%)", end='')
            
            # [CN]
            if nodes_in_layer:
                avg_neighbors = np.mean([len(self.graph[layer].get(n, [])) for n in nodes_in_layer])
                print(f"\n    [CN]: {avg_neighbors:.1f}")
    
    def _get_search_k(self, layer: int) -> int:
        """[CN]return[CN]"""
        base_k = self.config.efconstruction
        if layer == 0:
            return min(base_k * 4, len(self.vectors))  # Layer 0[CN]
        elif layer == 1:
            return min(base_k * 2, len(self.vectors))
        else:  # layer >= 2
            return min(base_k, len(self.vectors))
    
    def _select_neighbors_standard(self, node_id: int, candidates: List[Tuple[float, int]], 
                                  layer: int) -> List[int]:
        """
        [CN]HNSW[CN]
        [CN]、[CN]
        """
        if not candidates:
            return []
        
        # [CN]（[CN]3[CN]：0,1,2）
        if layer == 0:
            target_neighbors = min(self.M0, len(candidates))  # Layer 0: [CN]connect[CN] (M*2)
        elif layer == 1:
            target_neighbors = min(self.M // 2, len(candidates))  # Layer 1: M/2 connect[CN]
        else:  # layer == 2
            target_neighbors = min(self.M // 4, len(candidates))  # Layer 2: M/4 connect[CN]
        
        # [CN]
        candidates.sort()
        
        # [CN]，[CN]
        if layer >= 1:
            selected = []
            selected_set = set()
            
            for dist, neighbor in candidates:
                if len(selected) >= target_neighbors:
                    break
                
                # [CN]
                should_add = True
                if layer >= 2 and len(selected) > 0:
                    # [CN]
                    min_diversity_dist = dist * 0.5  # [CN]50%[CN]
                    for s in selected[:5]:  # [CN]
                        if self._distance(self.vectors[neighbor], self.vectors[s]) < min_diversity_dist:
                            should_add = False
                            break
                
                if should_add and neighbor not in selected_set:
                    selected.append(neighbor)
                    selected_set.add(neighbor)
            
            # [CN]，[CN]
            if len(selected) < target_neighbors // 2:
                for _, neighbor in candidates:
                    if neighbor not in selected_set:
                        selected.append(neighbor)
                        selected_set.add(neighbor)
                        if len(selected) >= target_neighbors:
                            break
            
            return selected
        else:
            # Layer 0: [CN]
            return [neighbor for _, neighbor in candidates[:target_neighbors]]
    
    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """calculateL2[CN]"""
        return np.sum((a - b) ** 2)
    
    def _print_stats(self):
        """print[CN]"""
        print("\n[CN]:")
        
        total_nodes = len(self.vectors)
        for layer in range(self.max_layers + 1):
            nodes_in_layer = [n for n, level in self.node_levels.items() if level >= layer]
            
            if nodes_in_layer:
                neighbor_counts = [len(self.graph[layer].get(n, [])) for n in nodes_in_layer]
                avg_neighbors = np.mean(neighbor_counts)
                min_neighbors = np.min(neighbor_counts)
                max_neighbors = np.max(neighbor_counts)
                
                print(f"  Layer {layer}: {len(nodes_in_layer)} [CN] "
                      f"({len(nodes_in_layer)/total_nodes*100:.1f}%)")
                print(f"    [CN] - [CN]: {avg_neighbors:.1f}, "
                      f"[CN]: {min_neighbors}, [CN]: {max_neighbors}")
            else:
                print(f"  Layer {layer}: 0 [CN]")
        
        print(f"\n  [CN]: node_{self.entry_point} ([CN]: {self.node_levels[self.entry_point]})")
    
    def save_trident_format(self, output_dir: str, dataset_name: str = "dataset"):
        """[CN]Trident[CN]"""
        print(f"\n[CN]Trident[CN]: {output_dir}")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        ntotal = len(self.vectors)
        d = self.vectors.shape[1]
        
        # [CN]
        node_file = f"{output_dir}/nodes.bin"
        with open(node_file, 'wb') as f:
            f.write(struct.pack('i', ntotal))
            f.write(struct.pack('i', d))
            f.write(struct.pack('i', self.entry_point))
            
            for i in range(ntotal):
                f.write(struct.pack('i', i))
                f.write(self.vectors[i].tobytes())
        
        print(f"  ✓ [CN]: {node_file} ({Path(node_file).stat().st_size/1024/1024:.1f} MB)")
        
        # [CN]
        neighbor_file = f"{output_dir}/neighbors.bin"
        num_levels = self.max_layers + 1
        maxM0 = self.M0  # [CN]M0[CN]
        
        with open(neighbor_file, 'wb') as f:
            f.write(struct.pack('i', ntotal))
            f.write(struct.pack('i', num_levels))
            f.write(struct.pack('i', maxM0))
            
            # [CN]
            for node_id in range(ntotal):
                for layer in range(num_levels):
                    f.write(struct.pack('i', node_id))
                    f.write(struct.pack('i', layer))
                    
                    # [CN]
                    neighbors = []
                    if node_id in self.node_levels and self.node_levels[node_id] >= layer:
                        neighbors = self.graph[layer].get(node_id, [])
                    
                    # [CN]maxM0[CN]，[CN]-1[CN]
                    padded = list(neighbors) + [-1] * (maxM0 - len(neighbors))
                    
                    for n in padded[:maxM0]:  # [CN]maxM0
                        f.write(struct.pack('i', n))
        
        print(f"  ✓ [CN]: {neighbor_file} ({Path(neighbor_file).stat().st_size/1024/1024:.1f} MB)")
        
        return node_file, neighbor_file


def read_fvecs(filename):
    """[CN]fvecs[CN]"""
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
    """[CN] - [CN]HNSW[CN]"""
    parser = argparse.ArgumentParser(description='[CN]HNSW[CN]')
    # [CN]Dataset[CN]
    available_configs = list_available_configs()
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=available_configs,
                       help=f'Dataset[CN]，[CN]: {available_configs}')
    parser.add_argument('--data-path', type=str, help='[CN]（fvecs[CN]）')
    parser.add_argument('--output-dir', type=str, help='[CN]（[CN]: ~/trident/dataset/Dataset[CN]/）')
    parser.add_argument('--seed', type=int, default=42, help='[CN]（[CN]: 42）')
    
    args = parser.parse_args()
    
    print("=== [CN]HNSW[CN] ===")
    print("[CN]HNSW[CN]\n")
    
    # [CN]Dataset[CN]
    config = get_config(args.dataset)
    print(f"[CN]Dataset[CN]: {args.dataset}")
    print(f"  Vector dimension: {config.vector_dimension}")
    print(f"  [CN]Number of documents: {config.num_docs:,}")
    print(f"  HNSW[CN]: M={config.M}, efConstruction={config.efconstruction}, layers={config.layer}")
    print()
    
    # [CN]
    if args.data_path:
        base_path = args.data_path
    else:
        # [CN]
        base_path = f"~/trident/dataset/{args.dataset}/base.fvecs"
    
    # [CN]
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"~/trident/dataset/{args.dataset}"
    
    # [CN]
    print(f"[CN]: {base_path}")
    vectors = read_fvecs(base_path)
    print(f"[CN]: {vectors.shape}")
    
    # [CN]
    if vectors.shape[1] != config.vector_dimension:
        print(f"[CN]: [CN]({vectors.shape[1]})[CN]({config.vector_dimension})[CN]")
    
    # create[CN]
    builder = FastStandardHNSWBuilder(config=config, seed=args.seed)
    
    # [CN]
    builder.build_index(vectors)
    
    # [CN]Trident[CN]
    builder.save_trident_format(output_dir, dataset_name=args.dataset)
    
    print("\n[CN]！")
    print(f"[CN]: {output_dir}/")
    print(f"  - nodes.bin: [CN]")
    print(f"  - neighbors.bin: [CN]")


if __name__ == "__main__":
    main()