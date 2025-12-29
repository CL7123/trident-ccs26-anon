#!/usr/bin/env python3
"""
内存消耗分析脚本
测量TridentSearcher的静态存储和动态内存使用
"""

import os
import sys
import json
from datetime import datetime

def measure_static_storage(dataset_name: str = "laion", base_path: str = "~/trident/dataset"):
    """测量静态文件存储大小"""
    dataset_path = os.path.join(base_path, dataset_name)
    
    results = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'file_sizes': {},
        'server_sizes': {},
        'total_storage': {}
    }
    
    # 基础文件
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
    
    # 服务器shares文件
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
    
    # 总计
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
    """生成与论文表格对比的结果"""
    
    # 提取关键数据
    embed_size_gb = static_results['file_sizes'].get('base.fvecs', {}).get('size_gb', 0)
    
    # Graph size = neighbors.bin + nodes.bin
    neighbors_gb = static_results['file_sizes'].get('neighbors.bin', {}).get('size_gb', 0)
    nodes_gb = static_results['file_sizes'].get('nodes.bin', {}).get('size_gb', 0)
    graph_size_gb = neighbors_gb + nodes_gb
    
    # 服务器shares大小
    server_sizes = []
    for i in [1, 2, 3]:
        server_gb = static_results['server_sizes'].get(f'server_{i}', {}).get('total_gb', 0)
        server_sizes.append(server_gb)
    
    # 输出对比表格
    print("\n" + "="*80)
    print("TridentSearcher 内存消耗对比分析")
    print("="*80)
    print(f"数据集: {static_results['dataset'].upper()}")
    print()
    
    print("| 组件 | 大小 (GB) | Server 1 | Server 2 | Server 3 | 总计 |")
    print("|------|-----------|----------|----------|----------|------|")
    print(f"| Embed. Size | {embed_size_gb:.3f} | - | - | - | {embed_size_gb:.3f} |")
    print(f"| Graph Size | {graph_size_gb:.3f} | - | - | - | {graph_size_gb:.3f} |")
    print(f"| VDPF Shares | - | {server_sizes[0]:.3f} | {server_sizes[1]:.3f} | {server_sizes[2]:.3f} | {sum(server_sizes):.3f} |")
    
    total_static = embed_size_gb + graph_size_gb + sum(server_sizes)
    print(f"| **静态总计** | **{total_static:.3f}** | **{server_sizes[0]:.3f}** | **{server_sizes[1]:.3f}** | **{server_sizes[2]:.3f}** | **{total_static:.3f}** |")
    
    print()
    print("与现有方法对比:")
    print("- Compass Server: ~2 GB (SH + Mal)")
    print("- HE-Cluster Server: ~8.75 GB")
    print(f"- TridentSearcher (单服务器): ~{(embed_size_gb + graph_size_gb + server_sizes[0]):.3f} GB")
    print(f"- TridentSearcher (三服务器总计): ~{total_static:.3f} GB")
    
    # 分析结果
    single_server_gb = embed_size_gb + graph_size_gb + server_sizes[0]
    print(f"\n性能优势:")
    print(f"- 相比 Compass: 单服务器节省 {max(0, 2 - single_server_gb):.3f} GB")
    print(f"- 相比 HE-Cluster: 单服务器节省 {max(0, 8.75 - single_server_gb):.3f} GB") 
    print(f"- 分布式分摊: 每个服务器只需 {single_server_gb:.3f} GB")
    
    # 保存到文件
    if output_file:
        with open(output_file, 'w') as f:
            f.write(f"# TridentSearcher 内存消耗分析报告\\n\\n")
            f.write(f"**数据集**: {static_results['dataset']}\\n")
            f.write(f"**生成时间**: {static_results['timestamp']}\\n\\n")
            
            f.write("## 静态存储需求\\n\\n")
            f.write("| 组件 | 大小 (GB) |\\n")
            f.write("|------|-----------|\\n")
            f.write(f"| 向量嵌入 | {embed_size_gb:.3f} |\\n")
            f.write(f"| 图结构 | {graph_size_gb:.3f} |\\n")
            f.write(f"| VDPF份额 (3服务器) | {sum(server_sizes):.3f} |\\n")
            f.write(f"| **总计** | **{total_static:.3f}** |\\n\\n")
            
            f.write("## 与现有方法对比\\n\\n")
            f.write("| 方法 | 单服务器内存 (GB) | 总内存 (GB) |\\n")
            f.write("|------|-------------------|-------------|\\n")
            f.write("| Compass | ~2.0 | ~2.0 |\\n")
            f.write("| HE-Cluster | ~8.75 | ~8.75 |\\n")
            f.write(f"| TridentSearcher | ~{single_server_gb:.3f} | ~{total_static:.3f} |\\n")
            
        print(f"\\n报告已保存到: {output_file}")

def main():
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    else:
        dataset = "laion"
    
    print(f"分析数据集: {dataset}")
    print("="*50)
    
    # 测量静态存储
    static_results = measure_static_storage(dataset)
    
    # 生成对比报告
    output_file = f"~/trident/experiment/{dataset}_memory_analysis.md"
    generate_comparison_table(static_results, output_file)
    
    # 保存原始数据
    json_file = f"~/trident/experiment/{dataset}_memory_data.json"
    with open(json_file, 'w') as f:
        json.dump(static_results, f, indent=2)
    
    print(f"原始数据已保存到: {json_file}")

if __name__ == "__main__":
    main()