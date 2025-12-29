#!/usr/bin/env python3

import sys
import os
import socket
import json
import numpy as np
import time
import concurrent.futures
import random
import argparse
import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('~/trident/src')

from dpf_wrapper import VDPFVectorWrapper
from basic_functionalities import get_config, MPC23SSS, Share
from share_data import DatasetLoader


class MultiprocessConfigurableTestClient:
    """测试多进程优化服务器的客户端"""
    
    def __init__(self, dataset: str = "laion"):
        self.dataset = dataset
        self.config = get_config(dataset)
        self.dpf_wrapper = VDPFVectorWrapper(dataset_name=dataset)
        self.mpc = MPC23SSS(self.config)
        
        # 预加载原始数据用于验证
        data_dir = f"~/trident/dataset/{dataset}"
        loader = DatasetLoader(data_dir)
        self.original_nodes, self.num_layers, self.max_neighbors = loader.load_neighbors()
        total_neighbor_entries = len(self.original_nodes) * self.num_layers
        print(f"预加载了 {len(self.original_nodes)} 个节点的邻居列表")
        print(f"  线性化后共 {total_neighbor_entries} 个邻居列表条目（{len(self.original_nodes)} 节点 × {self.num_layers} 层）")
        
        # 服务器地址
        self.servers = [
            ("192.168.50.21", 8001),
            ("192.168.50.22", 8002),
            ("192.168.50.23", 8003)
        ]
        
        self.connections = {}
        
    def connect_to_servers(self):
        """连接到所有多进程优化服务器"""
        for i, (host, port) in enumerate(self.servers):
            server_id = i + 1
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((host, port))
                self.connections[server_id] = sock
            except Exception:
                pass
        
    def _send_request(self, server_id: int, request: dict) -> dict:
        """向指定服务器发送请求"""
        sock = self.connections[server_id]
        
        request_data = json.dumps(request).encode()
        sock.sendall(len(request_data).to_bytes(4, 'big'))
        sock.sendall(request_data)
        
        length_bytes = sock.recv(4)
        length = int.from_bytes(length_bytes, 'big')
        data = b''
        while len(data) < length:
            chunk = sock.recv(min(length - len(data), 4096))
            data += chunk
        
        return json.loads(data.decode())
    
    def test_mmap_query(self, node_id: int = 1723, layer: int = 0):
        """测试多进程优化查询"""
        # 生成VDPF密钥 - 使用node_id*3 + layer计算邻居列表索引
        neighbor_index = node_id * 3 + layer
        print(f"\nDEBUG: 生成VDPF密钥 - node_id={node_id}, layer={layer}, neighbor_index={neighbor_index}")
        # 由于dpf_wrapper的generate_keys方法使用node_id参数作为alpha，我们直接传入neighbor_index
        keys = self.dpf_wrapper.generate_keys('neighbor', neighbor_index)
        
        # 并行查询所有服务器
        start_time = time.time()
        
        def query_server(server_id):
            request = {
                'command': 'query_node_vector',
                'dpf_key': keys[server_id - 1],
                'query_id': f'multiprocess_test_{int(time.time())}',
                'neighbor_index': neighbor_index,
                'node_id': node_id,
                'layer': layer
            }
            response = self._send_request(server_id, request)
            return server_id, response
        
        # 并行执行查询
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(query_server, sid) for sid in self.connections]
            results = {}
            
            for future in concurrent.futures.as_completed(futures):
                server_id, response = future.result()
                results[server_id] = response
        
        if all(r.get('status') == 'success' for r in results.values()):
            # 提取时间信息
            timings = {}
            for server_id, result in results.items():
                timing = result.get('timing', {})
                timings[server_id] = {
                    'phase1': timing.get('phase1_time', 0) / 1000,  # 转换为秒
                    'phase2': timing.get('phase2_time', 0) / 1000,
                    'phase3': timing.get('phase3_time', 0) / 1000,
                    'phase4': timing.get('phase4_time', 0) / 1000,
                    'total': timing.get('total', 0) / 1000
                }
            
            # 计算平均时间
            avg_timings = {}
            for phase in ['phase1', 'phase2', 'phase3', 'phase4', 'total']:
                avg_timings[phase] = np.mean([t[phase] for t in timings.values()])
            
            # 重构最终结果
            final_result = self._reconstruct_final_result(results)
            
            # DEBUG: 打印重构前的份额信息
            print(f"\nDEBUG: 重构前的份额信息:")
            for sid, result in results.items():
                if result.get('status') == 'success':
                    shares = result['result_share']
                    print(f"  Server {sid} 份额前5个值: {shares[:5]}")
            
            # 验证结果正确性并获取匹配度
            match_ratio = self._verify_result(node_id, layer, final_result)
            
            # 只打印核心信息
            print(f"\n查询结果 (节点{node_id}, 层{layer}):")
            print(f"  阶段1 (多进程VDPF评估): {avg_timings['phase1']:.2f}秒")
            print(f"  阶段2 (e/f计算): {avg_timings['phase2']:.2f}秒")
            print(f"  阶段3 (数据交换): {avg_timings['phase3']:.2f}秒")
            print(f"  阶段4 (重构): {avg_timings['phase4']:.2f}秒")
            print(f"  服务器内部总计: {avg_timings['total']:.2f}秒")
            if match_ratio is not None:
                print(f"  邻居匹配率: {match_ratio:.2%}")
            
            return avg_timings, final_result
            
        else:
            print("❌ 查询失败:")
            for server_id, result in results.items():
                if result.get('status') != 'success':
                    print(f"  MultiProcess Server {server_id}: {result.get('message', 'Unknown error')}")
            return None, None
    
    def _reconstruct_final_result(self, results):
        """重构最终结果 - 邻居列表"""
        # 获取至少两个服务器的响应
        server_ids = []
        for server_id in [1, 2, 3]:
            if server_id in results and results[server_id].get('status') == 'success':
                server_ids.append(server_id)
        
        if len(server_ids) < 2:
            print("错误：至少需要2个服务器的响应才能重构")
            return []
        
        # 使用前两个可用的服务器进行重构
        server_ids = sorted(server_ids)[:2]
        print(f"DEBUG: 使用服务器 {server_ids} 进行重构")
        
        # 动态获取邻居列表长度
        first_result = results[server_ids[0]]['result_share']
        num_neighbors = len(first_result)
        
        # 重构每个邻居ID
        reconstructed_neighbors = []
        
        for i in range(num_neighbors):
            shares = [
                Share(results[server_ids[0]]['result_share'][i], server_ids[0]),
                Share(results[server_ids[1]]['result_share'][i], server_ids[1])
            ]
            
            reconstructed = self.mpc.reconstruct(shares)
            
            # DEBUG: 打印重构值
            if i < 5:  # 只打印前5个
                print(f"  DEBUG: 位置{i} - 份额: [{shares[0].value}, {shares[1].value}], 重构值: {reconstructed}")
                
                # 测试：手动验证一个已知的值
                if i == 0:
                    # 创建一个测试秘密共享并重构
                    test_value = 4420  # 原始邻居列表的第一个值
                    test_shares = self.mpc.share_secret(test_value)
                    test_reconstructed = self.mpc.reconstruct([test_shares[0], test_shares[1]])
                    print(f"    TEST: 测试值 {test_value} -> 份额 [{test_shares[0].value}, {test_shares[1].value}] -> 重构 {test_reconstructed}")
            
            # 邻居ID是非负整数，不需要缩放
            reconstructed_neighbors.append(reconstructed)
        
        # DEBUG: 打印重构后的邻居列表信息
        print(f"\nDEBUG: 重构后的邻居列表信息:")
        print(f"  邻居数量: {len(reconstructed_neighbors)}")
        # 过滤掉填充值显示
        valid_neighbors = [n for n in reconstructed_neighbors if n != self.config.prime - 1 and n < len(self.original_nodes)]
        print(f"  有效邻居数量: {len(valid_neighbors)}")
        if valid_neighbors:
            print(f"  前10个有效邻居: {valid_neighbors[:10]}")
        
        return reconstructed_neighbors
    
    def _verify_result(self, node_id: int, layer: int, reconstructed_neighbors: list):
        """验证重构结果的正确性，返回匹配度"""
        try:
            if node_id < len(self.original_nodes) and layer < self.num_layers:
                original_neighbors = self.original_nodes[node_id][layer]
                
                # DEBUG: 打印原始邻居列表信息
                print(f"\nDEBUG: 原始邻居列表信息 (节点 {node_id}, 层 {layer}):")
                print(f"  邻居数量: {len(original_neighbors)}")
                print(f"  前10个邻居: {original_neighbors[:10]}")
                
                # 过滤掉填充值（-1或field_size-1）
                reconstructed_filtered = [n for n in reconstructed_neighbors if n != self.config.prime - 1 and n < len(self.original_nodes)]
                
                # 计算匹配度
                if len(original_neighbors) == 0:
                    return 1.0 if len(reconstructed_filtered) == 0 else 0.0
                
                # 计算交集
                matches = len(set(original_neighbors) & set(reconstructed_filtered))
                match_ratio = matches / len(original_neighbors)
                
                print(f"  重构邻居数量: {len(reconstructed_filtered)}")
                print(f"  匹配数量: {matches}/{len(original_neighbors)}")
                print(f"  匹配率: {match_ratio:.2%}")
                
                return match_ratio
            else:
                return None
                
        except Exception:
            return None
    
    def disconnect_from_servers(self):
        """断开所有服务器连接"""
        for sock in self.connections.values():
            sock.close()
        self.connections.clear()


def generate_markdown_report(dataset, query_details, avg_phases, avg_similarity):
    """生成Markdown格式的测试报告"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    markdown = f"""# 测试结果报告 - {dataset}

**生成时间**: {timestamp}  
**数据集**: {dataset}  
**查询次数**: {len(query_details)}

## 详细查询结果

| 查询编号 | 节点ID | 层 | 阶段1 (VDPF) | 阶段2 (e/f) | 阶段3 (交换) | 阶段4 (重构) | 总时间 | 邻居匹配率 |
|---------|--------|-----|--------------|-------------|--------------|--------------|--------|-----------|
"""
    
    for q in query_details:
        markdown += f"| {q['query_num']} | {q['node_id']} | {q.get('layer', 0)} | "
        markdown += f"{q['timings']['phase1']:.2f}s | "
        markdown += f"{q['timings']['phase2']:.2f}s | "
        markdown += f"{q['timings']['phase3']:.2f}s | "
        markdown += f"{q['timings']['phase4']:.2f}s | "
        markdown += f"{q['timings']['total']:.2f}s | "
        markdown += f"{q['similarity']:.2%} |\n"
    
    markdown += f"""
## 平均性能统计

- **阶段1 (多进程VDPF评估)**: {avg_phases['phase1']:.2f}秒
- **阶段2 (e/f计算)**: {avg_phases['phase2']:.2f}秒
- **阶段3 (数据交换)**: {avg_phases['phase3']:.2f}秒
- **阶段4 (重构)**: {avg_phases['phase4']:.2f}秒
- **服务器内部总计**: {avg_phases['total']:.2f}秒
- **平均邻居匹配率**: {avg_similarity:.2%}

## 性能分析

### 时间分布
"""
    
    # 计算各阶段时间占比
    total_avg = avg_phases['total']
    phase1_pct = (avg_phases['phase1'] / total_avg) * 100
    phase2_pct = (avg_phases['phase2'] / total_avg) * 100
    phase3_pct = (avg_phases['phase3'] / total_avg) * 100
    phase4_pct = (avg_phases['phase4'] / total_avg) * 100
    
    markdown += f"""
- 阶段1 (VDPF评估): {phase1_pct:.1f}%
- 阶段2 (e/f计算): {phase2_pct:.1f}%
- 阶段3 (数据交换): {phase3_pct:.1f}%
- 阶段4 (重构): {phase4_pct:.1f}%

### 吞吐量
- 平均查询时间: {total_avg:.2f}秒
- 理论吞吐量: {1/total_avg:.2f} 查询/秒
"""
    
    return markdown


def main():
    """主函数"""
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='向量级多进程优化客户端')
    parser.add_argument('--dataset', type=str, default='siftsmall', 
                        choices=['siftsmall', 'laion', 'tripclick', 'ms_marco', 'nfcorpus'],
                        help='数据集名称 (默认: siftsmall)')
    parser.add_argument('--num-queries', type=int, default=10,
                        help='测试查询数量 (默认: 10)')
    parser.add_argument('--no-report', action='store_true',
                        help='不保存测试报告')
    
    args = parser.parse_args()
    
    print(f"=== 多进程配置测试 - 数据集: {args.dataset} ===")
    
    client = MultiprocessConfigurableTestClient(args.dataset)
    
    try:
        client.connect_to_servers()
        
        if len(client.connections) == 0:
            print("错误：无法连接到任何多进程服务器")
            return
        
        all_timings = []
        all_similarities = []
        query_details = []  # 存储每次查询的详细信息
        
        # 获取节点总数
        total_nodes = len(client.original_nodes)
        
        # 随机选择节点
        random_nodes = random.sample(range(total_nodes), min(args.num_queries, total_nodes))
        
        print(f"将对 {len(random_nodes)} 个随机节点进行查询测试...\n")
        
        for i, node_id in enumerate(random_nodes):
            # 随机选择一个层进行测试
            layer = random.randint(0, 2)  # 假设有3层
            print(f"查询 {i+1}/{len(random_nodes)}: 节点 {node_id}, 层 {layer}")
            timings, final_result = client.test_mmap_query(node_id=node_id, layer=layer)
            
            if timings:
                all_timings.append(timings)
                # 获取匹配度
                match_ratio = client._verify_result(node_id, layer, final_result)
                if match_ratio is not None:
                    all_similarities.append(match_ratio)
                
                # 保存查询详情
                query_details.append({
                    'query_num': i + 1,
                    'node_id': node_id,
                    'layer': layer,
                    'timings': timings,
                    'similarity': match_ratio if match_ratio is not None else 0.0
                })
        
        # 计算平均值
        if all_timings:
            print(f"\n=== 平均性能统计 ({len(all_timings)} 个成功查询) ===")
            avg_phases = {}
            for phase in ['phase1', 'phase2', 'phase3', 'phase4', 'total']:
                avg_phases[phase] = np.mean([t[phase] for t in all_timings])
            
            print(f"  阶段1 (多进程VDPF评估): {avg_phases['phase1']:.2f}秒")
            print(f"  阶段2 (e/f计算): {avg_phases['phase2']:.2f}秒")
            print(f"  阶段3 (数据交换): {avg_phases['phase3']:.2f}秒")
            print(f"  阶段4 (重构): {avg_phases['phase4']:.2f}秒")
            print(f"  服务器内部总计: {avg_phases['total']:.2f}秒")
            
            if all_similarities:
                avg_similarity = np.mean(all_similarities)
                print(f"  平均邻居匹配率: {avg_similarity:.2%}")
            else:
                avg_similarity = 0.0
            
            # 保存报告（除非指定了--no-report）
            if not args.no_report and query_details:
                report_file = "~/trident/result.md"
                markdown_report = generate_markdown_report(
                    args.dataset, 
                    query_details, 
                    avg_phases,
                    avg_similarity
                )
                
                # 检查文件是否存在，如果存在则追加，否则创建新文件
                if os.path.exists(report_file):
                    # 追加模式，先添加分隔符
                    with open(report_file, 'a', encoding='utf-8') as f:
                        f.write("\n\n---\n\n")  # 添加分隔符
                        f.write(markdown_report)
                else:
                    # 创建新文件
                    with open(report_file, 'w', encoding='utf-8') as f:
                        f.write(markdown_report)
                
                print(f"\n测试报告已{'追加' if os.path.exists(report_file) else '保存'}到: {report_file}")
            
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.disconnect_from_servers()


if __name__ == "__main__":
    main()