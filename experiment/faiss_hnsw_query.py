import numpy as np
import faiss
import time
import argparse
import os
import sys
from typing import Tuple, Dict, Any

# Add parent directory to path to import domain_config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.domain_config import get_config, CONFIGS, SIFTSMALL, LAION, TRIPCLICK, NFCORPUS


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


def build_hnsw_index(data: np.ndarray, M: int = 32, ef_construction: int = 200) -> faiss.IndexHNSWFlat:
    """Build HNSW index with specified parameters."""
    print(f"Building HNSW index with M={M}, ef_construction={ef_construction}")
    d = data.shape[1]
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = ef_construction
    
    # Add vectors to index
    start_time = time.time()
    index.add(data)
    build_time = time.time() - start_time
    print(f"Index built in {build_time:.2f} seconds")
    print(f"Index contains {index.ntotal} vectors")
    
    return index, build_time


def search_hnsw(index: faiss.IndexHNSWFlat, queries: np.ndarray, k: int, ef_search: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Perform HNSW search with specified ef_search parameter."""
    index.hnsw.efSearch = ef_search
    
    start_time = time.time()
    distances, indices = index.search(queries, k)
    search_time = time.time() - start_time
    
    return distances, indices, search_time


def compute_recall(predictions: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Compute recall@k metric."""
    assert predictions.shape[0] == ground_truth.shape[0]
    
    total = 0
    for i in range(predictions.shape[0]):
        pred_set = set(predictions[i, :k])
        gt_set = set(ground_truth[i, :k])
        total += len(pred_set.intersection(gt_set))
    
    return total / (predictions.shape[0] * k)


def compute_mrr_at_k(predictions: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Compute MRR@k metric."""
    assert predictions.shape[0] == ground_truth.shape[0]
    
    mrr_sum = 0.0
    num_queries = predictions.shape[0]
    
    for i in range(num_queries):
        # Get the first k predictions
        pred = predictions[i, :k]
        # Get all ground truth (not just first k)
        gt = set(ground_truth[i])
        
        # Find the rank of the first correct prediction
        for rank, pred_id in enumerate(pred):
            if pred_id in gt:
                mrr_sum += 1.0 / (rank + 1)
                break
    
    return mrr_sum / num_queries if num_queries > 0 else 0.0


def run_experiment(dataset_path: str, dataset_name: str, M: int = 32, ef_construction: int = 200, 
                   ef_search_values: list = None, k_values: list = None) -> Dict[str, Any]:
    """Run HNSW experiments on a dataset."""
    if ef_search_values is None:
        ef_search_values = [50, 100, 200, 500]
    if k_values is None:
        k_values = [1, 10, 100]
    
    # Load data
    base_path = os.path.join(dataset_path, dataset_name, 'base.fvecs')
    query_path = os.path.join(dataset_path, dataset_name, 'query.fvecs')
    gt_path = os.path.join(dataset_path, dataset_name, 'gt.ivecs')
    
    print(f"\nLoading dataset: {dataset_name}")
    base_data = read_fvecs(base_path)
    query_data = read_fvecs(query_path)
    ground_truth = read_ivecs(gt_path)
    
    print(f"Base vectors: {base_data.shape}")
    print(f"Query vectors: {query_data.shape}")
    print(f"Ground truth: {ground_truth.shape}")
    
    # Build index
    index, build_time = build_hnsw_index(base_data, M, ef_construction)
    
    results = {
        'dataset': dataset_name,
        'base_vectors': base_data.shape[0],
        'dimensions': base_data.shape[1],
        'query_vectors': query_data.shape[0],
        'M': M,
        'ef_construction': ef_construction,
        'build_time': build_time,
        'experiments': []
    }
    
    # Run searches with different parameters
    for ef_search in ef_search_values:
        for k in k_values:
            if k > ground_truth.shape[1]:
                continue
                
            print(f"\nSearching with ef_search={ef_search}, k={k}")
            distances, indices, search_time = search_hnsw(index, query_data, k, ef_search)
            recall = compute_recall(indices, ground_truth, k)
            mrr = compute_mrr_at_k(indices, ground_truth, k)
            
            qps = query_data.shape[0] / search_time
            
            experiment = {
                'ef_search': ef_search,
                'k': k,
                'search_time': search_time,
                'qps': qps,
                'recall': recall,
                'mrr': mrr
            }
            results['experiments'].append(experiment)
            
            print(f"  Search time: {search_time:.3f}s")
            print(f"  QPS: {qps:.1f}")
            print(f"  Recall@{k}: {recall:.4f}")
            print(f"  MRR@{k}: {mrr:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='FAISS HNSW Query Tool with domain_config parameters')
    parser.add_argument('--dataset', type=str, default='siftsmall', 
                        choices=['siftsmall', 'nfcorpus', 'laion', 'tripclick'],
                        help='Dataset name')
    parser.add_argument('--dataset-path', type=str, default='~/trident/dataset',
                        help='Path to dataset directory')
    parser.add_argument('--use-config', action='store_true', default=True,
                        help='Use parameters from domain_config.py (default: True)')
    parser.add_argument('--M', type=int, help='Override HNSW M parameter')
    parser.add_argument('--ef-construction', type=int, help='Override HNSW ef_construction parameter')
    parser.add_argument('--ef-search', type=int, nargs='+',
                        help='Override HNSW ef_search values to test')
    parser.add_argument('--k', type=int, nargs='+', default=[1, 10, 100],
                        help='Number of nearest neighbors to retrieve')
    
    args = parser.parse_args()
    
    # Get parameters from domain_config if use_config is True
    if args.use_config:
        config = get_config(args.dataset)
        M = args.M if args.M is not None else config.M
        ef_construction = args.ef_construction if args.ef_construction is not None else config.efconstruction
        ef_search_values = args.ef_search if args.ef_search is not None else [config.efsearch]
        
        print(f"\nUsing domain_config parameters for {args.dataset}:")
        print(f"  M: {M} (config: {config.M})")
        print(f"  ef_construction: {ef_construction} (config: {config.efconstruction})")
        print(f"  ef_search: {ef_search_values} (config: {config.efsearch})")
        print(f"  Vector dimension: {config.vector_dimension}")
        print(f"  Expected docs: {config.num_docs}, queries: {config.num_queries}")
    else:
        M = args.M or 32
        ef_construction = args.ef_construction or 200
        ef_search_values = args.ef_search or [50, 100, 200, 500]
    
    # Run experiment
    results = run_experiment(
        args.dataset_path,
        args.dataset,
        M=M,
        ef_construction=ef_construction,
        ef_search_values=ef_search_values,
        k_values=args.k
    )
    
    # Print summary
    print("\n" + "="*60)
    print(f"HNSW Query Results Summary - {results['dataset']}")
    print("="*60)
    print(f"Dataset size: {results['base_vectors']} vectors, {results['dimensions']} dimensions")
    print(f"Index build time: {results['build_time']:.2f}s")
    print(f"HNSW parameters: M={results['M']}, ef_construction={results['ef_construction']}")
    print("\nQuery Performance:")
    print("-"*60)
    print(f"{'ef_search':<10} {'k':<5} {'QPS':<12} {'Recall':<10} {'MRR':<10} {'Time(s)':<10}")
    print("-"*60)
    
    for exp in results['experiments']:
        mrr_str = f"{exp.get('mrr', 0):.4f}" if 'mrr' in exp else "N/A"
        print(f"{exp['ef_search']:<10} {exp['k']:<5} {exp['qps']:<12.1f} {exp['recall']:<10.4f} {mrr_str:<10} {exp['search_time']:<10.3f}")


def run_all_datasets(dataset_path: str = '~/trident/dataset', k_values: list = None):
    """Run experiments on all datasets using their domain_config parameters."""
    if k_values is None:
        k_values = [1, 10, 100]
    
    all_results = {}
    
    for dataset_name in ['siftsmall', 'nfcorpus', 'laion', 'tripclick']:
        print(f"\n{'='*80}")
        print(f"Running experiments for {dataset_name.upper()}")
        print(f"{'='*80}")
        
        try:
            config = get_config(dataset_name)
            
            # Use config parameters with some variations in ef_search
            ef_search_values = [
                max(1, config.efsearch // 2),  # Half of config value
                config.efsearch,                # Config value
                config.efsearch * 2,            # Double config value
                config.efsearch * 4             # 4x config value
            ]
            
            results = run_experiment(
                dataset_path,
                dataset_name,
                M=config.M,
                ef_construction=config.efconstruction,
                ef_search_values=ef_search_values,
                k_values=k_values
            )
            
            all_results[dataset_name] = results
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            continue
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*80)
    
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name.upper()}:")
        print(f"  Dataset: {results['base_vectors']} vectors, {results['dimensions']}D")
        print(f"  HNSW: M={results['M']}, ef_construction={results['ef_construction']}")
        print(f"  Build time: {results['build_time']:.2f}s")
        
        # Find best MRR for each k
        best_by_k = {}
        for exp in results['experiments']:
            k = exp['k']
            mrr = exp.get('mrr', 0)
            if k not in best_by_k or mrr > best_by_k[k].get('mrr', 0):
                best_by_k[k] = exp
        
        print("  Best results by k:")
        for k in sorted(best_by_k.keys()):
            exp = best_by_k[k]
            mrr_str = f"{exp.get('mrr', 0):.4f}" if 'mrr' in exp else "N/A"
            print(f"    k={k}: MRR={mrr_str}, Recall={exp['recall']:.4f}, QPS={exp['qps']:.1f} (ef_search={exp['ef_search']})")
    
    return all_results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--run-all':
        # Run all datasets with their config parameters
        run_all_datasets()
    else:
        # Run single dataset with command line args
        main()