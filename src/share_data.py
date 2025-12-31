#!/usr/bin/env python3
"""
[CN]LAIONDataset[CN]
- [CN]：100k × 512[CN]
- [CN]：[CN] (node_id * 3 + layer)
"""

import sys
import os
import struct
import numpy as np
import json
import time
import argparse
from typing import Dict, Tuple

# [CN]
from domain_config import get_config as get_domain_config, list_available_configs, SIFTSMALL, LAION, TRIPCLICK, MS_MARCO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add parent directory to path for imports
from basic_functionalities import MPC23SSS, get_config as get_mpc_config


class DatasetLoader:
    """[CN]Dataset"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        
    def load_nodes(self) -> np.ndarray:
        """[CN]"""
        print("[CN]...")
        nodes_path = os.path.join(self.data_dir, "nodes.bin")
        
        with open(nodes_path, 'rb') as f:
            # [CN]header
            num_nodes = struct.unpack('<I', f.read(4))[0]
            vector_dim = struct.unpack('<I', f.read(4))[0]
            
            print(f"  [CN]: {num_nodes:,}")
            print(f"  Vector dimension: {vector_dim}")
            
            # [CN]header[CN]
            f.read(4)  # [CN]12[CN]
            
            # [CN]
            total_floats = num_nodes * vector_dim
            vectors = struct.unpack(f'<{total_floats}f', f.read(total_floats * 4))
            
            # [CN]
            nodes = np.array(vectors, dtype=np.float32).reshape(num_nodes, vector_dim)
            
        return nodes
    
    def load_neighbors(self) -> Tuple[np.ndarray, int, int]:
        """[CN]"""
        print("\n[CN]...")
        neighbors_path = os.path.join(self.data_dir, "neighbors.bin")
        
        with open(neighbors_path, 'rb') as f:
            # [CN]header
            num_nodes = struct.unpack('<I', f.read(4))[0]
            num_layers = struct.unpack('<I', f.read(4))[0]
            max_neighbors = struct.unpack('<I', f.read(4))[0]
            _ = struct.unpack('<I', f.read(4))[0]  # [CN]0
            
            print(f"  [CN]: {num_nodes:,}")
            print(f"  [CN]: {num_layers}")
            print(f"  [CN]: {max_neighbors}")
            
            # [CN]：[CN] (3[CN] × 128[CN] + 2[CN]) [CN]
            ints_per_node = num_layers * max_neighbors + 2
            
            # [CN]
            total_ints = num_nodes * ints_per_node
            print(f"  [CN] {total_ints:,} [CN]...")
            all_data = struct.unpack(f'<{total_ints}I', f.read(total_ints * 4))
            
            # [CN]
            neighbors = []
            
            for node_id in range(num_nodes):
                # calculate[CN]
                start_idx = node_id * ints_per_node
                
                node_neighbors = []
                for layer in range(num_layers):
                    # [CN]128[CN]
                    layer_start = start_idx + layer * max_neighbors
                    layer_data = all_data[layer_start:layer_start + max_neighbors]
                    
                    # [CN]（-1[CN]4294967295）
                    actual_neighbors = [x for x in layer_data if x != 4294967295 and x < num_nodes]
                    node_neighbors.append(actual_neighbors)
                
                neighbors.append(node_neighbors)
                
                # [CN]
                if (node_id + 1) % 10000 == 0:
                    print(f"  [CN]process {node_id + 1:,} / {num_nodes:,} [CN]")
            
            return neighbors, num_layers, max_neighbors


class DatasetSecretSharing:
    """[CN]Dataset[CN]"""
    
    def __init__(self, dataset_name: str):
        # [CN]
        self.domain_config = get_domain_config(dataset_name)
        
        # [CN]createMPC[CN]
        self.mpc_config = get_mpc_config(dataset_name)
        self.mpc = MPC23SSS(self.mpc_config)
        
        # [CN]
        self.field_size = self.domain_config.prime
        # [CN]
        # [CN]Dataset[CN]
        if dataset_name == "siftsmall":
            # siftsmall[CN][0, 180]，[CN]
            self.scale_factor = 2 ** 20  # [CN]180[CN]
        elif dataset_name == "tripclick":
            # tripclick[CN][-5, 5]，[CN]
            self.scale_factor = 2 ** 20  # 1M[CN]，[CN]768[CN]calculate[CN]
        elif dataset_name == "nfcorpus":
            # nfcorpus[CN][-12, 4]，[CN]
            self.scale_factor = 2 ** 22  # 4M[CN]，[CN]
        else:
            # [CN]Dataset（[CN]LAION）[CN][-1, 1]，[CN]
            self.scale_factor = 2 ** (self.domain_config.output_bits - 2)
        
    def float_to_field(self, value: float) -> int:
        """[CN]"""
        # [CN]，[CN]field_size[CN]
        # Python[CN]process[CN]，[CN][0, field_size)[CN]
        scaled = int(value * self.scale_factor)
        return scaled % self.field_size
    
    def share_nodes(self, nodes: np.ndarray) -> Dict[int, np.ndarray]:
        """[CN]"""
        num_nodes, vector_dim = nodes.shape
        print(f"\n[CN]...")
        print(f"  [CN]: {nodes.shape}")
        
        # initialize[CN]
        server_shares = {
            1: np.zeros((num_nodes, vector_dim), dtype=np.uint32),
            2: np.zeros((num_nodes, vector_dim), dtype=np.uint32),
            3: np.zeros((num_nodes, vector_dim), dtype=np.uint32)
        }
        
        # [CN]
        start_time = time.time()
        
        for i in range(num_nodes):
            # [CN]
            for j in range(vector_dim):
                # [CN]
                field_value = self.float_to_field(nodes[i, j])
                
                # [CN]
                shares = self.mpc.share_secret(field_value)
                
                # allocate[CN]
                for server_id in range(1, 4):
                    server_shares[server_id][i, j] = shares[server_id-1].value
            
            # [CN]
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (num_nodes - i - 1) / rate
                print(f"  [CN]: {i+1:,}/{num_nodes:,} ({(i+1)/num_nodes*100:.1f}%) - "
                      f"[CN]: {rate:.0f} [CN]/[CN] - [CN]: {eta:.0f}[CN]")
        
        return server_shares
    
    def share_neighbors_linearized(self, neighbors: list, num_layers: int, 
                                   max_neighbors: int) -> Dict[int, np.ndarray]:
        """[CN]"""
        num_nodes = len(neighbors)
        total_entries = num_nodes * num_layers
        
        print(f"\n[CN]...")
        print(f"  [CN]: {total_entries:,} × {max_neighbors}")
        
        # initialize[CN]
        server_shares = {
            1: np.zeros((total_entries, max_neighbors), dtype=np.uint32),
            2: np.zeros((total_entries, max_neighbors), dtype=np.uint32),
            3: np.zeros((total_entries, max_neighbors), dtype=np.uint32)
        }
        
        # [CN]
        start_time = time.time()
        processed = 0
        
        for node_id in range(num_nodes):
            for layer in range(num_layers):
                # calculate[CN]
                linear_idx = node_id * num_layers + layer
                
                # [CN]
                neighbor_list = neighbors[node_id][layer] if layer < len(neighbors[node_id]) else []
                
                # Padding[CN]max_neighbors，[CN]-1[CN]
                padding_value = self.field_size - 1  # -1 [CN]
                padded_neighbors = neighbor_list + [padding_value] * (max_neighbors - len(neighbor_list))
                
                # [CN]ID[CN]
                for j in range(max_neighbors):
                    neighbor_id = padded_neighbors[j]
                    
                    # [CN]
                    shares = self.mpc.share_secret(neighbor_id)
                    
                    # allocate[CN]
                    for server_id in range(1, 4):
                        server_shares[server_id][linear_idx, j] = shares[server_id-1].value
                
                processed += 1
                
                # [CN]
                if processed % 10000 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed
                    eta = (total_entries - processed) / rate
                    print(f"  [CN]: {processed:,}/{total_entries:,} ({processed/total_entries*100:.1f}%) - "
                          f"[CN]: {rate:.0f} [CN]/[CN] - [CN]: {eta:.0f}[CN]")
        
        return server_shares


def main():
    """[CN]：[CN]"""
    # [CN]
    parser = argparse.ArgumentParser(description='[CN]Dataset[CN]')
    available_configs = list_available_configs()
    parser.add_argument('--dataset', type=str, default=SIFTSMALL,
                        choices=available_configs,
                        help=f'Dataset[CN]，[CN]: {available_configs}')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Dataset[CN]')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='[CN]')
    args = parser.parse_args()
    
    print(f"=== Dataset[CN]: {args.dataset} ===\n")
    
    # [CN]
    domain_config = get_domain_config(args.dataset)
    print(f"[CN]Dataset[CN]: {args.dataset}")
    print(f"  Vector dimension: {domain_config.vector_dimension}")
    print(f"  [CN]Number of documents: {domain_config.num_docs:,}")
    print(f"  [CN]: p = {domain_config.prime:,}")
    print(f"  [CN]: {domain_config.output_bits}")
    print()
    
    # [CN]
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
    
    # 1. [CN]
    loader = DatasetLoader(data_dir)
    
    # [CN]
    nodes = loader.load_nodes()
    
    # [CN]
    neighbors, num_layers, max_neighbors = loader.load_neighbors()
    
    # 2. [CN]
    sharer = DatasetSecretSharing(args.dataset)
    
    # [CN]
    node_shares = sharer.share_nodes(nodes)
    
    # [CN]
    neighbor_shares = sharer.share_neighbors_linearized(neighbors, num_layers, max_neighbors)
    
    # 3. [CN]
    
    print("\n[CN]...")
    for server_id in range(1, 4):
        server_dir = os.path.join(output_dir, f"server_{server_id}")
        os.makedirs(server_dir, exist_ok=True)
        
        # [CN]
        nodes_path = os.path.join(server_dir, "nodes_shares.npy")
        np.save(nodes_path, node_shares[server_id])
        print(f"  Server {server_id} [CN]: {nodes_path}")
        
        # [CN]
        neighbors_path = os.path.join(server_dir, "neighbors_shares.npy")
        np.save(neighbors_path, neighbor_shares[server_id])
        print(f"  Server {server_id} [CN]: {neighbors_path}")
        
        # [CN]
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
        print(f"  Server {server_id} [CN]: {metadata_path}")
    
    print(f"\n✅ {args.dataset}Dataset[CN]！")
    print(f"\n[CN]: {output_dir}")
    print("[CN]:")
    print("  python share_data.py --dataset siftsmall")
    print("  python share_data.py --dataset laion")
    print("  python share_data.py --dataset tripclick --output-dir /custom/path")


if __name__ == "__main__":
    main()