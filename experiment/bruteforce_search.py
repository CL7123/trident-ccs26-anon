import numpy as np
import faiss
import time
import struct
import os
import sys
from typing import Tuple, Dict, Any

# Add parent directory to path to import domain_config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.domain_config import get_config, SIFTSMALL, LAION, TRIPCLICK, NFCORPUS


def read_fvecs(filename: str) -> np.ndarray:
    """Read .fvecs file format used by SIFT datasets."""
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.int32, count=1)
        dim = data[0]
        f.seek(0)
        data = np.fromfile(f, dtype=np.float32)
        data = data.reshape(-1, dim + 1)
        return data[:, 1:].copy()


def read_ivecs(filename: str) -> np.ndarray:
    """Read .ivecs file format for ground truth."""
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.int32, count=1)
        dim = data[0]
        f.seek(0)
        data = np.fromfile(f, dtype=np.int32)
        data = data.reshape(-1, dim + 1)
        return data[:, 1:].copy()


def compute_mrr_at_k(predictions: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Compute MRR@k metric."""
    assert predictions.shape[0] == ground_truth.shape[0]
    
    mrr_sum = 0.0
    num_queries = predictions.shape[0]
    
    for i in range(num_queries):
        pred = predictions[i, :k]
        gt = set(ground_truth[i])
        
        for rank, pred_id in enumerate(pred):
            if pred_id in gt:
                mrr_sum += 1.0 / (rank + 1)
                break
    
    return mrr_sum / num_queries if num_queries > 0 else 0.0


def run_bruteforce_search(base_data: np.ndarray, query_data: np.ndarray, k: int = 10, 
                         measure_individual: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run brute-force search using FAISS IndexFlatL2.
    
    Args:
        base_data: Base vectors to search in
        query_data: Query vectors
        k: Number of nearest neighbors
        measure_individual: If True, measure each query individually for accurate latency
    """
    d = base_data.shape[1]
    
    # Limit FAISS to single thread for more realistic comparison
    # This disables multi-threading optimizations to get closer to actual O(n*d) complexity
    original_threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    
    # Create brute-force index
    index = faiss.IndexFlatL2(d)
    
    # Add vectors to index
    index.add(base_data)
    
    # Warm up
    for _ in range(5):
        _, _ = index.search(query_data[:1], k)
    
    if measure_individual:
        # Measure individual query latency
        all_distances = []
        all_indices = []
        total_time = 0
        
        for i in range(len(query_data)):
            start_time = time.time()
            dist, idx = index.search(query_data[i:i+1], k)
            total_time += time.time() - start_time
            all_distances.append(dist[0])
            all_indices.append(idx[0])
        
        distances = np.array(all_distances)
        indices = np.array(all_indices)
        search_time = total_time
    else:
        # Batch search (faster but less accurate latency measurement)
        start_time = time.time()
        distances, indices = index.search(query_data, k)
        search_time = time.time() - start_time
    
    # Restore original thread count
    faiss.omp_set_num_threads(original_threads)
    
    return distances, indices, search_time


def test_dataset(dataset_name: str, dataset_path: str = '~/trident/dataset', k: int = 10):
    """Test brute-force search on a specific dataset."""
    # Get dataset configuration
    config = get_config(dataset_name)
    
    # Construct paths
    base_path = os.path.join(dataset_path, dataset_name)
    base_file = os.path.join(base_path, 'base.fvecs')
    query_file = os.path.join(base_path, 'query.fvecs')
    gt_file = os.path.join(base_path, 'gt.ivecs')
    
    # Check if files exist
    if not all(os.path.exists(f) for f in [base_file, query_file, gt_file]):
        print(f"Error: Missing files for {dataset_name}")
        return None
    
    # Load data
    print(f"\nLoading {dataset_name} dataset...")
    base_data = read_fvecs(base_file)
    query_data = read_fvecs(query_file)
    ground_truth = read_ivecs(gt_file)
    
    print(f"  Base vectors: {base_data.shape}")
    print(f"  Query vectors: {query_data.shape}")
    print(f"  Ground truth: {ground_truth.shape}")
    
    # Test different numbers of queries to understand scaling
    test_sizes = [10, 100, min(1000, len(query_data))]
    
    results = {
        'dataset': dataset_name,
        'base_vectors': base_data.shape[0],
        'dimensions': base_data.shape[1],
        'total_queries': query_data.shape[0],
        'tests': []
    }
    
    for num_queries in test_sizes:
        if num_queries > len(query_data):
            continue
            
        print(f"\n  Testing with {num_queries} queries:")
        
        # Select subset of queries
        query_subset = query_data[:num_queries]
        gt_subset = ground_truth[:num_queries]
        
        # Run brute-force search (with individual query measurement for accurate latency)
        distances, indices, search_time = run_bruteforce_search(base_data, query_subset, k, measure_individual=True)
        
        # Compute metrics
        mrr = compute_mrr_at_k(indices, gt_subset, k)
        avg_latency_ms = (search_time / num_queries) * 1000
        qps = num_queries / search_time
        
        print(f"    Total time: {search_time:.3f}s")
        print(f"    Average latency: {avg_latency_ms:.2f}ms")
        print(f"    QPS: {qps:.1f}")
        print(f"    MRR@{k}: {mrr:.4f}")
        
        results['tests'].append({
            'num_queries': num_queries,
            'search_time': search_time,
            'avg_latency_ms': avg_latency_ms,
            'qps': qps,
            'mrr': mrr
        })
    
    return results


def main():
    """Test brute-force search on all datasets."""
    print("="*60)
    print("Brute-Force Search Performance Test")
    print("(Single-threaded FAISS IndexFlatL2)")
    print("="*60)
    
    datasets = ['siftsmall', 'nfcorpus', 'laion', 'tripclick']
    all_results = {}
    
    for dataset in datasets:
        results = test_dataset(dataset)
        if results:
            all_results[dataset] = results
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY - Brute-Force Search Results")
    print("="*60)
    
    # Create a markdown table for results
    print("\n| Dataset | Vectors | Dims | Queries | Latency (ms) | QPS | MRR@10 |")
    print("|---------|---------|------|---------|--------------|-----|--------|")
    
    for dataset, results in all_results.items():
        # Use the 100-query test as reference (or the largest available)
        ref_test = None
        for test in results['tests']:
            if test['num_queries'] == 100:
                ref_test = test
                break
        if not ref_test and results['tests']:
            ref_test = results['tests'][-1]
        
        if ref_test:
            print(f"| {dataset.upper()} | {results['base_vectors']:,} | {results['dimensions']} | "
                  f"{ref_test['num_queries']} | {ref_test['avg_latency_ms']:.2f} | "
                  f"{ref_test['qps']:.1f} | {ref_test['mrr']:.4f} |")
    
    # Save detailed results
    output_file = '~/trident/experiment/bruteforce_results.txt'
    with open(output_file, 'w') as f:
        f.write("Brute-Force Search Detailed Results\n")
        f.write("="*60 + "\n\n")
        
        for dataset, results in all_results.items():
            f.write(f"\n{dataset.upper()}:\n")
            f.write(f"  Base vectors: {results['base_vectors']:,}\n")
            f.write(f"  Dimensions: {results['dimensions']}\n")
            f.write(f"  Test results:\n")
            
            for test in results['tests']:
                f.write(f"    {test['num_queries']} queries: ")
                f.write(f"latency={test['avg_latency_ms']:.2f}ms, ")
                f.write(f"QPS={test['qps']:.1f}, ")
                f.write(f"MRR@10={test['mrr']:.4f}\n")
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # Also create data for plotting
    plot_data = {}
    for dataset, results in all_results.items():
        # Use 100-query test for consistency
        for test in results['tests']:
            if test['num_queries'] == 100:
                plot_data[dataset] = {
                    'latency_ms': test['avg_latency_ms'],
                    'mrr': test['mrr']
                }
                break
    
    print("\nData for plotting (100 queries):")
    for dataset, data in plot_data.items():
        print(f"  '{dataset.upper()}': ({data['latency_ms']:.2f}, {data['mrr']:.4f}),")


if __name__ == "__main__":
    main()