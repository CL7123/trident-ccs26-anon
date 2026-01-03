
#!/usr/bin/env python3
"""
Test and fix TridentSearcher search algorithm
"""

import numpy as np
import heapq
import time
import struct
import argparse
import os
from typing import List, Tuple
from datetime import datetime

from domain_config import get_config, list_available_configs, SIFTSMALL, LAION, TRIPCLICK, MS_MARCO

class TridentSearcher:
    """Trident searcher"""

    def __init__(self, node_file: str, neighbor_file: str, config=None):
        """Initialize searcher"""
        self.config = config
        self.vectors = None
        self.graph = {}
        self.node_levels = {}
        self.entry_point = 0

        self._load_nodes(node_file)
        self._load_neighbors(neighbor_file)

        print(f"TridentSearcher initialized:")
        print(f"  Vectors: {len(self.vectors)}")
        print(f"  Dimension: {self.vectors.shape[1]}")
        print(f"  Layers: {len(self.graph)}")
        print(f"  Entry point: {self.entry_point}")
        if self.config:
            print(f"  Config efSearch: {self.config.efsearch}")
            print(f"  Expected documents: {self.config.num_docs:,}")

    def _load_nodes(self, filename: str):
        """Load node vectors"""
        with open(filename, 'rb') as f:
            ntotal = struct.unpack('i', f.read(4))[0]
            d = struct.unpack('i', f.read(4))[0]
            self.entry_point = struct.unpack('i', f.read(4))[0]
            
            vectors = []
            for i in range(ntotal):
                node_id = struct.unpack('i', f.read(4))[0]
                vector = np.frombuffer(f.read(d * 4), dtype=np.float32)
                vectors.append(vector)
            
            self.vectors = np.array(vectors)
    
    def _load_neighbors(self, filename: str):
        """Load neighbor relationships"""
        with open(filename, 'rb') as f:
            ntotal = struct.unpack('i', f.read(4))[0]
            num_levels = struct.unpack('i', f.read(4))[0]
            maxM0 = struct.unpack('i', f.read(4))[0]

            for layer in range(num_levels):
                self.graph[layer] = {}

            for i in range(ntotal):
                for layer in range(num_levels):
                    node_id = struct.unpack('i', f.read(4))[0]
                    layer_id = struct.unpack('i', f.read(4))[0]

                    neighbors = []
                    for _ in range(maxM0):
                        n = struct.unpack('i', f.read(4))[0]
                        if n >= 0:
                            neighbors.append(n)

                    if neighbors:
                        self.graph[layer][node_id] = neighbors

                        if node_id not in self.node_levels:
                            self.node_levels[node_id] = layer
                        else:
                            self.node_levels[node_id] = max(self.node_levels[node_id], layer)

    def search_single(self, query: np.ndarray, k: int, ef: int = None, return_stats: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Single query search

        Args:
            query: query vector
            k: number of nearest neighbors to return
            ef: efSearch parameter
            return_stats: whether to return statistics

        Returns:
            distances, indices or (distances, indices, stats)
        """
        # Use config efSearch as default value
        if ef is None:
            ef = self.config.efsearch if self.config else 32

        # Initialize statistics
        self.nodes_visited = set()
        self.neighborlists_accessed = 0

        # 1. Start from entry point, greedy search on higher layers
        curr_nearest = self.entry_point
        curr_level = self._get_node_level(self.entry_point)

        # Search down from highest layer to Layer 1
        for level in range(curr_level, 0, -1):
            nearest = self._greedy_search_layer(query, curr_nearest, 1, level)
            if nearest:
                curr_nearest = nearest[0]

        # 2. Extended search on Layer 0
        candidates = self._search_layer(query, [curr_nearest], ef, 0)

        # 3. Extract top-k results
        distances, indices = self._get_top_k(candidates, k)

        if return_stats:
            stats = {
                'nodes_visited': len(self.nodes_visited),
                'neighborlists_accessed': self.neighborlists_accessed
            }
            return distances, indices, stats

        return distances, indices
    
    # def _greedy_search_layer(self, query: np.ndarray, entry_point: int,
    #                         num_closest: int, layer: int) -> List[int]:
    #     """Greedy search on specified layer, return only num_closest nearest nodes"""
    #     visited = set()
    #     candidates = []  # (distance, node_id)
    #     W = []  # result set
    #
    #     # initialize
    #     visited.add(entry_point)
    #     d = self._distance(query, self.vectors[entry_point])
    #     heapq.heappush(candidates, (d, entry_point))
    #     W.append((d, entry_point))
    #
    #     while candidates:
    #         curr_dist, curr = heapq.heappop(candidates)
    #
    #         # If current node is farther than nearest in result set, stop search
    #         if curr_dist > W[0][0]:
    #             break
    #
    #         # check neighbors
    #         neighbors = self._get_neighbors(curr, layer)
    #         for neighbor in neighbors:
    #             if neighbor not in visited:
    #                 visited.add(neighbor)
    #                 d = self._distance(query, self.vectors[neighbor])
    #
    #                 if d < W[0][0] or len(W) < num_closest:
    #                     heapq.heappush(candidates, (d, neighbor))
    #                     W.append((d, neighbor))
    #                     W.sort(key=lambda x: x[0])
    #                     if len(W) > num_closest:
    #                         W.pop()
    #
    #     return [node for dist, node in W[:num_closest]]

    def _greedy_search_layer(self, query: np.ndarray, entry_point: int,
                            num_closest: int, layer: int) -> List[int]:
        """True greedy search - fast navigation on higher layers"""
        current = entry_point
        current_dist = self._distance(query, self.vectors[current])

        # Record visited nodes
        self.nodes_visited.add(current)

        # Record visited path (for debugging)
        path = [current]

        # True greedy: continuously find closer neighbors
        improved = True
        while improved:
            improved = False

            # Get neighbors of current node
            neighbors = self._get_neighbors(current, layer)
            self.neighborlists_accessed += 1  # Statistics for neighborlist access

            # Greedy selection: move immediately upon finding first closer neighbor
            for neighbor in neighbors:
                self.nodes_visited.add(neighbor)  # Record visited neighbors
                neighbor_dist = self._distance(query, self.vectors[neighbor])

                if neighbor_dist < current_dist:
                    # Found closer node, move immediately
                    current = neighbor
                    current_dist = neighbor_dist
                    path.append(current)
                    improved = True
                    break  # Critical: stop checking other neighbors immediately

        # Return final reached node
        return [current]
    
    def _search_layer(self, query: np.ndarray, entry_points: List[int],
                              ef: int, layer: int) -> List[Tuple[int, float]]:
        """Layer search algorithm"""
        visited = set()
        candidates = []  # min heap
        W = []  # result set, kept sorted

        # Initialize
        for point in entry_points:
            if point not in visited:
                visited.add(point)
                self.nodes_visited.add(point)  # Statistics for visited nodes
                d = self._distance(query, self.vectors[point])
                heapq.heappush(candidates, (d, point))
                W.append((d, point))

        # Main loop
        while candidates:
            curr_dist, curr = heapq.heappop(candidates)

            # Dynamic early stopping condition
            if W:
                W.sort(key=lambda x: x[0])
                if len(W) >= ef and curr_dist > W[ef-1][0]:
                    break

            # Explore neighbors
            neighbors = self._get_neighbors(curr, layer)
            self.neighborlists_accessed += 1  # Statistics for neighborlist access

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    self.nodes_visited.add(neighbor)  # Statistics for visited nodes
                    d = self._distance(query, self.vectors[neighbor])

                    # More flexible inclusion condition
                    if len(W) < ef or d < W[-1][0]:
                        heapq.heappush(candidates, (d, neighbor))
                        W.append((d, neighbor))

                        # Keep W sorted and limit size
                        if len(W) > ef:
                            W.sort(key=lambda x: x[0])
                            W = W[:ef]

        # Return final result
        W.sort(key=lambda x: x[0])
        return [(node, dist) for dist, node in W]

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate squared L2 distance"""
        return np.sum((a - b) ** 2)

    def _get_neighbors(self, node: int, layer: int) -> List[int]:
        """Get neighbors of node at specified layer"""
        if layer in self.graph and node in self.graph[layer]:
            return self.graph[layer][node]
        return []

    def _get_node_level(self, node: int) -> int:
        """Get highest layer level of node"""
        return self.node_levels.get(node, 0)

    def _get_top_k(self, candidates: List[Tuple[int, float]], k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Extract top-k results from candidate set"""
        distances = np.full(k, float('inf'), dtype=np.float32)
        indices = np.full(k, -1, dtype=np.int32)

        for i in range(min(k, len(candidates))):
            indices[i] = candidates[i][0]
            distances[i] = candidates[i][1]

        return distances, indices

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

def read_ivecs(filename):
    """Read ivecs format ground truth"""
    ivecs = []
    with open(filename, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec = struct.unpack('i' * dim, f.read(4 * dim))
            ivecs.append(vec)
    return np.array(ivecs)

def calculate_mrr_at_k(results, ground_truth, k=10):
    """Calculate MRR@k"""
    mrr_sum = 0.0
    num_queries = min(len(results), len(ground_truth))

    for i in range(num_queries):
        result = results[i][:k]
        gt = ground_truth[i]

        for rank, retrieved_id in enumerate(result):
            if retrieved_id in gt[:k]:
                mrr_sum += 1.0 / (rank + 1)
                break

    return mrr_sum / num_queries if num_queries > 0 else 0.0

def main():
    """Test improved search algorithm"""
    # Prepare result file, use timestamp to avoid overwrite
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f"~/trident/result_{timestamp}.md"
    results_content = []

    # Set command line parameters
    parser = argparse.ArgumentParser(description='Test HNSW search algorithm')
    available_configs = list_available_configs()
    parser.add_argument('--dataset', type=str, default=SIFTSMALL,
                        choices=available_configs,
                        help=f'Dataset name, options: {available_configs}')
    parser.add_argument('--num-queries', type=int, default=100,
                        help='Number of test queries (default: 100)')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of nearest neighbors to return (default: 10)')
    args = parser.parse_args()

    print(f"=== Test Improved Search Algorithm - Dataset: {args.dataset} ===\n")

    # Start recording results
    results_content.append(f"# Trident Search Results\n")
    results_content.append(f"**Test time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    results_content.append(f"**Dataset**: {args.dataset}\n")

    # Get dataset configuration
    config = get_config(args.dataset)
    print(f"Using dataset configuration: {args.dataset}")
    print(f"  Vector dimension: {config.vector_dimension}")
    print(f"  Expected documents: {config.num_docs:,}")
    print(f"  efSearch: {config.efsearch}")
    print()

    # Record configuration information
    results_content.append(f"\n## Configuration\n")
    results_content.append(f"- Vector dimension: {config.vector_dimension}\n")
    results_content.append(f"- Documents: {config.num_docs:,}\n")
    results_content.append(f"- HNSW parameters: M={config.M}, efConstruction={config.efconstruction}\n")
    results_content.append(f"- Default efSearch: {config.efsearch}\n")
    results_content.append(f"- Queries: {args.num_queries}\n")
    results_content.append(f"- k: {args.k}\n")

    # Build data path (consistent with index-builder.py output)
    base_path = f"~/trident/dataset/{args.dataset}"

    # Check if path exists
    if not os.path.exists(base_path):
        print(f"Error: dataset path does not exist: {base_path}")
        print(f"Available dataset directories:")
        dataset_dir = "~/trident/dataset"
        if os.path.exists(dataset_dir):
            for d in os.listdir(dataset_dir):
                if os.path.isdir(os.path.join(dataset_dir, d)):
                    print(f"  - {d}")
        return

    # Load data
    query_file = f"{base_path}/query.fvecs"
    gt_file = f"{base_path}/gt.ivecs"
    node_file = f"{base_path}/nodes.bin"
    neighbor_file = f"{base_path}/neighbors.bin"

    # Check if required files exist
    required_files = [query_file, gt_file, node_file, neighbor_file]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print(f"Error: missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        return

    queries = read_fvecs(query_file)
    ground_truth = read_ivecs(gt_file)

    print(f"Query data: {queries.shape}")
    print(f"Ground truth: {ground_truth.shape}")
    print(f"Node file: {node_file}")
    print(f"Neighbor file: {neighbor_file}")
    print(f"\nTesting current version...")

    # Record data information
    results_content.append(f"\n## Data Information\n")
    results_content.append(f"- Query vectors: {queries.shape}\n")
    results_content.append(f"- Ground truth: {ground_truth.shape}\n")

    # Create searcher
    searcher = TridentSearcher(node_file, neighbor_file, config=config)

    # Record test results
    results_content.append(f"\n## Test Results\n")
    results_content.append(f"| efSearch | MRR@{args.k} | Avg Latency (ms) | Avg Nodes Visited | Avg Neighborlists Accessed |\n")
    results_content.append(f"|----------|-----------|---------------|----------------|--------------------|\n")

    # Test different ef values (dynamically set based on configuration)
    base_ef = config.efsearch
    test_efs = [base_ef // 2, base_ef, base_ef * 2, base_ef * 4]
    test_efs = [ef for ef in test_efs if ef > 0]  # Filter out non-positive values

    for ef in test_efs:
        print(f"\n  efSearch = {ef}:")

        # Warmup
        for _ in range(5):
            _ = searcher.search_single(queries[0], args.k, ef)

        # Test
        all_results = []
        all_stats = []
        start_time = time.time()

        num_queries = min(args.num_queries, len(queries))
        for i in range(num_queries):
            _, indices, stats = searcher.search_single(queries[i], args.k, ef, return_stats=True)
            all_results.append(indices)
            all_stats.append(stats)

        elapsed = time.time() - start_time

        # Calculate MRR
        mrr = calculate_mrr_at_k(all_results, ground_truth[:num_queries], k=args.k)
        avg_latency = elapsed / num_queries * 1000

        # Calculate average visit statistics
        avg_nodes = np.mean([s['nodes_visited'] for s in all_stats])
        avg_neighborlists = np.mean([s['neighborlists_accessed'] for s in all_stats])

        print(f"    MRR@{args.k}: {mrr:.4f}")
        print(f"    Avg latency: {avg_latency:.2f}ms")
        print(f"    Avg nodes visited: {avg_nodes:.1f}")
        print(f"    Avg neighborlists accessed: {avg_neighborlists:.1f}")

        # Record result
        results_content.append(f"| {ef} | {mrr:.4f} | {avg_latency:.2f} | {avg_nodes:.1f} | {avg_neighborlists:.1f} |\n")

    # Add summary with best result
    results_content.append(f"\n## Summary\n")
    results_content.append(f"- Test completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    results_content.append(f"- Dataset: {args.dataset}\n")
    results_content.append(f"- Index location: {base_path}\n")

    # Print result content for manual saving
    print(f"\n{'='*60}")
    print("Test results (can be manually copied and saved):")
    print('='*60)
    for line in results_content:
        print(line.rstrip())
    print('='*60)

if __name__ == "__main__":
    main()