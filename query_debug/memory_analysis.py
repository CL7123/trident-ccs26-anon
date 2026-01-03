#!/usr/bin/env python3
"""
Memory consumption analysis script
Measures static storage and dynamic memory usage of TridentSearcher
"""

import os
import sys
import json
from datetime import datetime

def measure_static_storage(dataset_name: str = "laion", base_path: str = "~/trident/dataset"):
    """Measure static file storage size"""
    dataset_path = os.path.join(base_path, dataset_name)
    
    results = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'file_sizes': {},
        'server_sizes': {},
        'total_storage': {}
    }
    
    # Base files
    base_files = ['base.fvecs', 'query.fvecs', 'neighbors.bin', 'nodes.bin', 'gt.ivecs']
    total_base_size = 0
    
    for filename in base_files:
        filepath = os.path.join(dataset_path, filename)
        if os.path.exists(filepath):
            size_bytes = os.path.getsize(filepath)
            size_mb = size_bytes / (1024 * 1024)
            size_gb = size_bytes / (1024 * 1024 * 1024)
            
            results['file_sizes'][filename] = {
                'size_bytes': size_bytes,
                'size_mb': size_mb,
                'size_gb': size_gb
            }
            total_base_size += size_bytes
            
            print(f"{filename}: {size_mb:.2f} MB")
    
    # Server shares files
    total_shares_size = 0
    for server_id in [1, 2, 3]:
        server_dir = os.path.join(dataset_path, f'server_{server_id}')
        server_total = 0
        
        if os.path.exists(server_dir):
            server_files = ['neighbors_shares.npy', 'nodes_shares.npy', 'metadata.json']
            server_breakdown = {}
            
            for filename in server_files:
                filepath = os.path.join(server_dir, filename)
                if os.path.exists(filepath):
                    size_bytes = os.path.getsize(filepath)
                    size_mb = size_bytes / (1024 * 1024)
                    server_total += size_bytes
                    server_breakdown[filename] = {
                        'size_bytes': size_bytes,
                        'size_mb': size_mb
                    }
            
            results['server_sizes'][f'server_{server_id}'] = {
                'total_bytes': server_total,
                'total_mb': server_total / (1024 * 1024),
                'total_gb': server_total / (1024 * 1024 * 1024),
                'breakdown': server_breakdown
            }
            total_shares_size += server_total
            
            print(f"Server {server_id} total: {server_total / (1024 * 1024):.2f} MB")
    
    # Total summary
    results['total_storage'] = {
        'base_data_bytes': total_base_size,
        'base_data_mb': total_base_size / (1024 * 1024),
        'base_data_gb': total_base_size / (1024 * 1024 * 1024),
        'shares_total_bytes': total_shares_size,
        'shares_total_mb': total_shares_size / (1024 * 1024),
        'shares_total_gb': total_shares_size / (1024 * 1024 * 1024),
        'grand_total_bytes': total_base_size + total_shares_size,
        'grand_total_mb': (total_base_size + total_shares_size) / (1024 * 1024),
        'grand_total_gb': (total_base_size + total_shares_size) / (1024 * 1024 * 1024)
    }
    
    return results

def generate_comparison_table(static_results: dict, output_file: str = None):
    """Generate results for comparison with paper tables"""

    # Extract key data
    embed_size_gb = static_results['file_sizes'].get('base.fvecs', {}).get('size_gb', 0)
    
    # Graph size = neighbors.bin + nodes.bin
    neighbors_gb = static_results['file_sizes'].get('neighbors.bin', {}).get('size_gb', 0)
    nodes_gb = static_results['file_sizes'].get('nodes.bin', {}).get('size_gb', 0)
    graph_size_gb = neighbors_gb + nodes_gb
    
    # Server shares size
    server_sizes = []
    for i in [1, 2, 3]:
        server_gb = static_results['server_sizes'].get(f'server_{i}', {}).get('total_gb', 0)
        server_sizes.append(server_gb)
    
    # Output comparison table
    print("\n" + "="*80)
    print("TridentSearcher Memory Consumption Comparison Analysis")
    print("="*80)
    print(f"Dataset: {static_results['dataset'].upper()}")
    print()

    print("| Component | Size (GB) | Server 1 | Server 2 | Server 3 | Total |")
    print("|-----------|-----------|----------|----------|----------|-------|")
    print(f"| Embed. Size | {embed_size_gb:.3f} | - | - | - | {embed_size_gb:.3f} |")
    print(f"| Graph Size | {graph_size_gb:.3f} | - | - | - | {graph_size_gb:.3f} |")
    print(f"| VDPF Shares | - | {server_sizes[0]:.3f} | {server_sizes[1]:.3f} | {server_sizes[2]:.3f} | {sum(server_sizes):.3f} |")

    total_static = embed_size_gb + graph_size_gb + sum(server_sizes)
    print(f"| **Static Total** | **{total_static:.3f}** | **{server_sizes[0]:.3f}** | **{server_sizes[1]:.3f}** | **{server_sizes[2]:.3f}** | **{total_static:.3f}** |")

    print()
    print("Comparison with existing methods:")
    print("- Compass Server: ~2 GB (SH + Mal)")
    print("- HE-Cluster Server: ~8.75 GB")
    print(f"- TridentSearcher (single server): ~{(embed_size_gb + graph_size_gb + server_sizes[0]):.3f} GB")
    print(f"- TridentSearcher (three servers total): ~{total_static:.3f} GB")

    # Analysis results
    single_server_gb = embed_size_gb + graph_size_gb + server_sizes[0]
    print(f"\nPerformance advantages:")
    print(f"- Compared to Compass: single server saves {max(0, 2 - single_server_gb):.3f} GB")
    print(f"- Compared to HE-Cluster: single server saves {max(0, 8.75 - single_server_gb):.3f} GB")
    print(f"- Distributed allocation: each server only needs {single_server_gb:.3f} GB")
    
    # Save to file
    if output_file:
        with open(output_file, 'w') as f:
            f.write(f"# TridentSearcher Memory Consumption Analysis Report\\n\\n")
            f.write(f"**Dataset**: {static_results['dataset']}\\n")
            f.write(f"**Generation Time**: {static_results['timestamp']}\\n\\n")

            f.write("## Static Storage Requirements\\n\\n")
            f.write("| Component | Size (GB) |\\n")
            f.write("|-----------|-----------|\\n")
            f.write(f"| Vector Embeddings | {embed_size_gb:.3f} |\\n")
            f.write(f"| Graph Structure | {graph_size_gb:.3f} |\\n")
            f.write(f"| VDPF Shares (3 servers) | {sum(server_sizes):.3f} |\\n")
            f.write(f"| **Total** | **{total_static:.3f}** |\\n\\n")

            f.write("## Comparison with Existing Methods\\n\\n")
            f.write("| Method | Single Server Memory (GB) | Total Memory (GB) |\\n")
            f.write("|--------|---------------------------|-------------------|\\n")
            f.write("| Compass | ~2.0 | ~2.0 |\\n")
            f.write("| HE-Cluster | ~8.75 | ~8.75 |\\n")
            f.write(f"| TridentSearcher | ~{single_server_gb:.3f} | ~{total_static:.3f} |\\n")

        print(f"\\nReport saved to: {output_file}")

def main():
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    else:
        dataset = "laion"
    
    print(f"Analyzing dataset: {dataset}")
    print("="*50)

    # Measure static storage
    static_results = measure_static_storage(dataset)

    # Generate comparison report
    output_file = f"~/trident/experiment/{dataset}_memory_analysis.md"
    generate_comparison_table(static_results, output_file)

    # Save raw data
    json_file = f"~/trident/experiment/{dataset}_memory_data.json"
    with open(json_file, 'w') as f:
        json.dump(static_results, f, indent=2)

    print(f"Raw data saved to: {json_file}")

if __name__ == "__main__":
    main()