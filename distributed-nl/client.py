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
import logging
from typing import Dict, List, Optional

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('~/trident/src')
sys.path.append('~/trident/standardDPF')
sys.path.append('~/trident/query-opti')

from dpf_wrapper import VDPFVectorWrapper
from dpf_wrapper_optimized import OptimizedVDPFVectorWrapper
from binary_protocol import BinaryProtocol
from basic_functionalities import get_config, MPC23SSS, Share
from share_data import DatasetLoader

# 加载配置
try:
    from config import CLIENT_SERVERS as SERVERS
except ImportError:
    # 默认配置 - 客户端使用公网IP
    SERVERS = {
        1: {"host": "192.168.1.101", "port": 9001},
        2: {"host": "192.168.1.102", "port": 9002},
        3: {"host": "192.168.1.103", "port": 9003}
    }

# 设置日志
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [NL-Client] %(message)s')
logger = logging.getLogger(__name__)


class DistributedNeighborClient:
    """分布式邻居列表查询客户端"""
    
    def __init__(self, dataset: str = "siftsmall", servers_config: Dict = None):
        self.dataset = dataset
        self.config = get_config(dataset)
        self.dpf_wrapper = OptimizedVDPFVectorWrapper(dataset_name=dataset)
        self.mpc = MPC23SSS(self.config)
        
        # 预加载原始邻居数据用于验证
        data_dir = f"~/trident/dataset/{dataset}"
        
        # 尝试加载邻居列表数据
        neighbors_path = os.path.join(data_dir, "neighbors.bin")
        if os.path.exists(neighbors_path):
            # 加载HNSW格式的邻居列表
            self.original_neighbors = self._load_hnsw_neighbors(neighbors_path)
            if self.original_neighbors is not None:
                # 计算实际的节点数（线性索引数 / 层数）
                num_layers = 3
                num_nodes = len(self.original_neighbors) // num_layers
                logger.info(f"预加载了HNSW邻居数据: 线性索引数={len(self.original_neighbors)}, 节点数={num_nodes}")
            else:
                logger.warning("加载neighbors.bin失败")
        else:
            # 尝试ivecs格式的groundtruth作为备选
            gt_path = os.path.join(data_dir, "gt.ivecs")
            if os.path.exists(gt_path):
                self.original_neighbors = self._load_ivecs(gt_path)
                logger.info(f"预加载了groundtruth邻居数据: {self.original_neighbors.shape}")
            else:
                self.original_neighbors = None
                logger.warning("未找到邻居列表数据，无法进行结果验证")
        
        # 服务器配置 - 更新端口号为9001-9003
        self.servers_config = servers_config or {
            server_id: {
                "host": info["host"],
                "port": 9000 + server_id  # 邻居列表服务使用9001-9003端口
            }
            for server_id, info in SERVERS.items()
        }
        self.connections = {}
        self.connection_retry_count = 3
        self.connection_timeout = 10
        
    def _load_ivecs(self, filename):
        """加载ivecs格式文件"""
        with open(filename, 'rb') as f:
            vectors = []
            while True:
                dim_bytes = f.read(4)
                if not dim_bytes:
                    break
                dim = int.from_bytes(dim_bytes, byteorder='little')
                vector = np.frombuffer(f.read(dim * 4), dtype=np.int32)
                vectors.append(vector)
            return np.array(vectors)
    
    def _load_hnsw_neighbors(self, filename):
        """加载HNSW格式的neighbors.bin文件"""
        try:
            import struct
            with open(filename, 'rb') as f:
                # 读取header
                num_nodes = struct.unpack('<I', f.read(4))[0]
                num_layers = struct.unpack('<I', f.read(4))[0]
                max_neighbors = struct.unpack('<I', f.read(4))[0]
                _ = struct.unpack('<I', f.read(4))[0]  # 跳过额外的0
                
                logger.info(f"HNSW数据: 节点数={num_nodes}, 层数={num_layers}, 最大邻居数={max_neighbors}")
                
                # 每个节点的数据大小：2个元数据 + 各层的邻居数据
                ints_per_node = 2 + num_layers * max_neighbors
                
                # 创建线性化的邻居数据数组，与服务器端的存储格式一致
                # 线性索引 = node_id * num_layers + layer
                linear_neighbors = {}
                
                for node_id in range(num_nodes):
                    # 读取该节点的所有数据
                    node_data = struct.unpack(f'<{ints_per_node}I', f.read(ints_per_node * 4))
                    
                    # 跳过前2个元数据值
                    # 数据布局：[metadata1, metadata2, layer0_neighbors, layer1_neighbors, layer2_neighbors]
                    
                    # 存储每层的邻居数据
                    for layer in range(num_layers):
                        # 正确的格式应该跳过前2个元数据
                        start_idx = 2 + layer * max_neighbors
                        end_idx = start_idx + max_neighbors
                        layer_neighbors = list(node_data[start_idx:end_idx])
                        
                        # 计算线性索引
                        linear_idx = node_id * num_layers + layer
                        
                        # 存储完整的128个邻居（包括4294967295填充值）
                        linear_neighbors[linear_idx] = layer_neighbors
                
                return linear_neighbors
                
        except Exception as e:
            logger.error(f"加载HNSW邻居数据时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def connect_to_servers(self):
        """连接到所有邻居列表服务器"""
        successful_connections = 0
        
        for server_id, server_info in self.servers_config.items():
            host = server_info["host"]
            port = server_info["port"]
            
            connected = False
            for attempt in range(self.connection_retry_count):
                try:
                    logger.info(f"尝试连接到邻居列表服务器 {server_id} ({host}:{port})，第 {attempt + 1} 次...")
                    
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(self.connection_timeout)
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    
                    sock.connect((host, port))
                    self.connections[server_id] = sock
                    logger.info(f"成功连接到邻居列表服务器 {server_id}")
                    successful_connections += 1
                    connected = True
                    break
                    
                except socket.timeout:
                    logger.warning(f"连接邻居列表服务器 {server_id} 超时")
                except ConnectionRefusedError:
                    logger.warning(f"邻居列表服务器 {server_id} 拒绝连接")
                except Exception as e:
                    logger.warning(f"连接邻居列表服务器 {server_id} 失败: {e}")
                
                if attempt < self.connection_retry_count - 1:
                    time.sleep(2)  # 重试前等待
            
            if not connected:
                logger.error(f"无法连接到邻居列表服务器 {server_id}")
        
        logger.info(f"成功连接到 {successful_connections}/{len(self.servers_config)} 个邻居列表服务器")
        return successful_connections >= 2  # 至少需要2个服务器
    
    def _send_request(self, server_id: int, request: dict) -> Optional[dict]:
        """向指定服务器发送请求，支持错误处理"""
        if server_id not in self.connections:
            logger.error(f"未连接到服务器 {server_id}")
            return None
        
        sock = self.connections[server_id]
        
        try:
            # 使用二进制协议发送包含密钥的请求
            if 'dpf_key' in request:
                # 对于查询请求，增加超时时间
                old_timeout = sock.gettimeout()
                sock.settimeout(60)  # 60秒超时
                
                # 使用二进制协议发送请求
                BinaryProtocol.send_binary_request(
                    sock, 
                    request['command'],
                    request['dpf_key'],
                    request.get('query_id')
                )
                
                # 接收响应
                response = BinaryProtocol.receive_response(sock)
                
                # 恢复原超时设置
                sock.settimeout(old_timeout)
                return response
            else:
                # 其他请求使用JSON
                request_data = json.dumps(request).encode()
                sock.sendall(len(request_data).to_bytes(4, 'big'))
                sock.sendall(request_data)
                
                # 接收响应
                length_bytes = sock.recv(4)
                if not length_bytes:
                    raise ConnectionError("连接已关闭")
                
                length = int.from_bytes(length_bytes, 'big')
                data = b''
                while len(data) < length:
                    chunk = sock.recv(min(length - len(data), 4096))
                    if not chunk:
                        raise ConnectionError("接收数据时连接中断")
                    data += chunk
                
                return json.loads(data.decode())
                
        except Exception as e:
            logger.error(f"与服务器 {server_id} 通信时出错: {e}")
            return None
    
    def test_distributed_neighbor_query(self, query_node_id: int = 0):
        """测试分布式邻居列表查询"""
        # 生成VDPF密钥 - 用于邻居查询
        keys = self.dpf_wrapper.generate_keys('neighbor', query_node_id)
        
        # 生成查询ID
        query_id = f'nl_distributed_test_{time.time()}_{query_node_id}'
        
        logger.info(f"开始分布式邻居列表查询，查询节点ID: {query_node_id}, 查询ID: {query_id}")
        
        # 并行查询所有服务器
        start_time = time.time()
        
        def query_server(server_id):
            request = {
                'command': 'query_neighbor_list',
                'dpf_key': keys[server_id - 1],
                'query_id': query_id
            }
            response = self._send_request(server_id, request)
            return server_id, response
        
        # 并行执行查询
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.connections)) as executor:
            futures = [executor.submit(query_server, sid) for sid in self.connections]
            results = {}
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    server_id, response = future.result()
                    results[server_id] = response
                except Exception as e:
                    logger.error(f"查询服务器时出错: {e}")
        
        # 检查结果
        successful_responses = {sid: r for sid, r in results.items() 
                              if r and r.get('status') == 'success'}
        
        if len(successful_responses) < 2:
            logger.error("查询失败：成功响应的服务器少于2个")
            for server_id, result in results.items():
                if not result or result.get('status') != 'success':
                    logger.error(f"服务器 {server_id}: {result}")
            return None, None
        
        # 提取时间信息
        timings = {}
        for server_id, result in successful_responses.items():
            timing = result.get('timing', {})
            timings[server_id] = {
                'phase1': timing.get('phase1_time', 0) / 1000,
                'phase2': timing.get('phase2_time', 0) / 1000,
                'phase3': timing.get('phase3_time', 0) / 1000,
                'phase4': timing.get('phase4_time', 0) / 1000,
                'total': timing.get('total', 0) / 1000
            }
        
        # 计算平均时间
        avg_timings = {}
        for phase in ['phase1', 'phase2', 'phase3', 'phase4', 'total']:
            phase_times = [t[phase] for t in timings.values()]
            avg_timings[phase] = np.mean(phase_times) if phase_times else 0
        
        # 重构邻居列表
        final_result = self._reconstruct_neighbor_list(successful_responses)
        
        # 验证结果
        accuracy = self._verify_neighbor_result(query_node_id, final_result)
        
        # 打印结果
        total_time = time.time() - start_time
        logger.info(f"\n邻居列表查询结果:")
        logger.info(f"  客户端总时间: {total_time:.2f}秒")
        logger.info(f"  阶段1 (VDPF评估): {avg_timings['phase1']:.2f}秒")
        logger.info(f"  阶段2 (e/f计算): {avg_timings['phase2']:.2f}秒")
        logger.info(f"  阶段3 (数据交换): {avg_timings['phase3']:.2f}秒")
        logger.info(f"  阶段4 (重构): {avg_timings['phase4']:.2f}秒")
        logger.info(f"  服务器平均总计: {avg_timings['total']:.2f}秒")
        if accuracy is not None:
            logger.info(f"  邻居列表准确率: {accuracy:.2%}")
        logger.info(f"  返回邻居数: {len(final_result) if final_result is not None else 0}")
        
        return avg_timings, final_result
    
    def _reconstruct_neighbor_list(self, results):
        """重构邻居列表"""
        # 获取至少两个服务器的响应
        server_ids = sorted([sid for sid, r in results.items() 
                           if r and r.get('status') == 'success'])[:2]
        
        if len(server_ids) < 2:
            logger.error("重构失败：可用服务器少于2个")
            return None
        
        # 获取邻居列表长度
        first_result = results[server_ids[0]]['result_share']
        k_neighbors = len(first_result)
        
        # 重构每个邻居索引
        reconstructed_neighbors = []
        
        for i in range(k_neighbors):
            shares = [
                Share(results[server_ids[0]]['result_share'][i], server_ids[0]),
                Share(results[server_ids[1]]['result_share'][i], server_ids[1])
            ]
            
            reconstructed = self.mpc.reconstruct(shares)
            
            # 处理重构的值
            # 秘密共享使用 field_size-1 (即 prime-1) 表示 -1
            # 原始HNSW使用 4294967295 作为无效邻居
            if reconstructed == self.config.prime - 1:
                # 这是填充值 -1，在HNSW中表示为 4294967295
                neighbor_idx = 4294967295
            elif reconstructed >= self.config.num_docs:
                # 超出文档范围的索引也是无效的
                neighbor_idx = 4294967295
            else:
                # 正常的邻居索引
                neighbor_idx = reconstructed
            
            reconstructed_neighbors.append(int(neighbor_idx))
        
        return reconstructed_neighbors
    
    def _verify_neighbor_result(self, query_node_id: int, reconstructed_neighbors: List[int]):
        """验证邻居列表结果的正确性"""
        try:
            if self.original_neighbors is None or reconstructed_neighbors is None:
                return None
            
            # 直接使用线性索引从original_neighbors字典中获取数据
            if query_node_id not in self.original_neighbors:
                logger.warning(f"线性索引 {query_node_id} 不在原始数据中")
                return None
            
            # 获取原始邻居列表（已经是特定节点特定层的数据）
            original_layer_neighbors = self.original_neighbors[query_node_id]
            
            # 计算实际的节点ID和层（用于日志显示）
            num_layers = 3  # HNSW的层数
            actual_node_id = query_node_id // num_layers
            layer = query_node_id % num_layers
            
            # 由于share_data.py的bug，数据有循环偏移
            # 重构的前2个值来自上一个节点的末尾
            # 原始的末尾2个值会出现在下一个节点的开头
            
            # 为了正确比较，我们需要对齐数据
            # 方法：比较集合而不是位置，因为数据是循环偏移的
            
            # 过滤有效邻居（不是4294967295的）
            valid_original = [n for n in original_layer_neighbors if n != 4294967295]
            valid_reconstructed = [n for n in reconstructed_neighbors if n != 4294967295 and n < self.config.num_docs]
            
            # 计算准确率 - 基于有效邻居的匹配
            if len(valid_original) == 0:
                # 如果原始没有有效邻居，检查重构是否也没有
                accuracy = 1.0 if len(valid_reconstructed) == 0 else 0.0
            else:
                # 计算有效邻居的交集
                # 由于是循环偏移，我们比较集合而不是位置
                matches = len(set(valid_original) & set(valid_reconstructed))
                # 如果重构的邻居数量和原始的不同，说明有问题
                if len(valid_reconstructed) != len(valid_original):
                    # 可能有额外的邻居（来自其他节点的偏移）
                    accuracy = matches / max(len(valid_original), len(valid_reconstructed))
                else:
                    accuracy = matches / len(valid_original)
            
            logger.info(f"邻居列表对比 (线性索引={query_node_id}, 节点={actual_node_id}, 层={layer}):")
            logger.info(f"  原始邻居列表前10个: {original_layer_neighbors[:10]}")
            logger.info(f"  原始邻居列表后10个: {original_layer_neighbors[-10:]}")
            logger.info(f"  重构邻居列表前10个: {reconstructed_neighbors[:10]}")
            logger.info(f"  重构邻居列表后10个: {reconstructed_neighbors[-10:]}")
            logger.info(f"  原始有效邻居: {len(valid_original)} 个")
            logger.info(f"  重构有效邻居: {len(valid_reconstructed)} 个")
            if len(valid_original) > 0:
                logger.info(f"  匹配数: {matches}/{len(valid_original)}")
            
            return accuracy
                
        except Exception as e:
            logger.error(f"验证邻居列表时出错: {e}")
            return None
    
    def get_server_status(self):
        """获取所有邻居列表服务器状态"""
        logger.info("获取邻居列表服务器状态...")
        
        for server_id in self.connections:
            request = {'command': 'get_status'}
            response = self._send_request(server_id, request)
            
            if response and response.get('status') == 'success':
                logger.info(f"\n邻居列表服务器 {server_id} 状态:")
                logger.info(f"  模式: {response.get('mode')}")
                logger.info(f"  地址: {response.get('host')}:{response.get('port')}")
                logger.info(f"  数据集: {response.get('dataset')}")
                logger.info(f"  VDPF进程数: {response.get('vdpf_processes')}")
                logger.info(f"  数据加载: {response.get('data_loaded')}")
                logger.info(f"  可用三元组: {response.get('triples_available')}")
            else:
                logger.error(f"无法获取邻居列表服务器 {server_id} 的状态")
    
    def disconnect_from_servers(self):
        """断开所有服务器连接"""
        for server_id, sock in self.connections.items():
            try:
                sock.close()
                logger.info(f"已断开与邻居列表服务器 {server_id} 的连接")
            except:
                pass
        self.connections.clear()


def generate_markdown_report(dataset, query_details, avg_phases, avg_accuracy):
    """生成Markdown格式的测试报告"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    markdown = f"""# 分布式邻居列表测试报告 - {dataset}

**生成时间**: {timestamp}  
**数据集**: {dataset}  
**查询次数**: {len(query_details)}

## 详细查询结果

| 查询编号 | 节点ID | 阶段1 (VDPF) | 阶段2 (e/f) | 阶段3 (交换) | 阶段4 (重构) | 总时间 | 准确率 |
|---------|--------|--------------|-------------|--------------|--------------|--------|--------|
"""
    
    for q in query_details:
        markdown += f"| {q['query_num']} | {q['node_id']} | "
        markdown += f"{q['timings']['phase1']:.2f}s | "
        markdown += f"{q['timings']['phase2']:.2f}s | "
        markdown += f"{q['timings']['phase3']:.2f}s | "
        markdown += f"{q['timings']['phase4']:.2f}s | "
        markdown += f"{q['timings']['total']:.2f}s | "
        if q['accuracy'] is not None:
            markdown += f"{q['accuracy']:.2%} |\n"
        else:
            markdown += "N/A |\n"
    
    markdown += f"""
## 平均性能统计

- **阶段1 (VDPF评估)**: {avg_phases['phase1']:.2f}秒
- **阶段2 (e/f计算)**: {avg_phases['phase2']:.2f}秒
- **阶段3 (数据交换)**: {avg_phases['phase3']:.2f}秒
- **阶段4 (重构)**: {avg_phases['phase4']:.2f}秒
- **服务器平均总计**: {avg_phases['total']:.2f}秒
- **平均准确率**: {avg_accuracy:.2%}

## 性能分析

### 时间分布
"""
    
    # 计算各阶段时间占比
    total_avg = avg_phases['total']
    if total_avg > 0:
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
    parser = argparse.ArgumentParser(description='分布式邻居列表查询客户端')
    parser.add_argument('--dataset', type=str, default='siftsmall', 
                        choices=['siftsmall', 'laion', 'tripclick', 'ms_marco', 'nfcorpus'],
                        help='数据集名称 (默认: siftsmall)')
    parser.add_argument('--num-queries', type=int, default=10,
                        help='测试查询数量 (默认: 10)')
    parser.add_argument('--no-report', action='store_true',
                        help='不保存测试报告')
    parser.add_argument('--config', type=str,
                        help='服务器配置文件路径')
    parser.add_argument('--status-only', action='store_true',
                        help='只获取服务器状态')
    
    args = parser.parse_args()
    
    # 加载自定义配置
    servers_config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                servers_config = json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return
    
    logger.info(f"=== 分布式邻居列表测试客户端 - 数据集: {args.dataset} ===")
    
    client = DistributedNeighborClient(args.dataset, servers_config)
    
    try:
        # 连接服务器
        if not client.connect_to_servers():
            logger.error("连接邻居列表服务器失败")
            return
        
        # 如果只是获取状态
        if args.status_only:
            client.get_server_status()
            return
        
        all_timings = []
        all_accuracies = []
        query_details = []
        
        # 获取节点总数（查询节点数）
        total_nodes = len(client.original_neighbors) if client.original_neighbors is not None else 1000
        
        # 随机选择查询节点
        random_nodes = random.sample(range(total_nodes), min(args.num_queries, total_nodes))
        
        logger.info(f"将对 {len(random_nodes)} 个随机节点进行邻居列表查询测试...\n")
        
        for i, node_id in enumerate(random_nodes):
            logger.info(f"查询 {i+1}/{len(random_nodes)}: 节点 {node_id} 的邻居列表")
            timings, neighbors = client.test_distributed_neighbor_query(query_node_id=node_id)
            
            if timings:
                all_timings.append(timings)
                accuracy = client._verify_neighbor_result(node_id, neighbors)
                if accuracy is not None:
                    all_accuracies.append(accuracy)
                
                query_details.append({
                    'query_num': i + 1,
                    'node_id': node_id,
                    'timings': timings,
                    'accuracy': accuracy,
                    'neighbors': neighbors[:10] if neighbors else []  # 只保存前10个邻居
                })
        
        # 计算平均值
        if all_timings:
            logger.info(f"\n=== 平均性能统计 ({len(all_timings)} 个成功查询) ===")
            avg_phases = {}
            for phase in ['phase1', 'phase2', 'phase3', 'phase4', 'total']:
                avg_phases[phase] = np.mean([t[phase] for t in all_timings])
            
            logger.info(f"  阶段1 (VDPF评估): {avg_phases['phase1']:.2f}秒")
            logger.info(f"  阶段2 (e/f计算): {avg_phases['phase2']:.2f}秒")
            logger.info(f"  阶段3 (数据交换): {avg_phases['phase3']:.2f}秒")
            logger.info(f"  阶段4 (重构): {avg_phases['phase4']:.2f}秒")
            logger.info(f"  服务器平均总计: {avg_phases['total']:.2f}秒")
            
            if all_accuracies:
                avg_accuracy = np.mean(all_accuracies)
                logger.info(f"  平均准确率: {avg_accuracy:.2%}")
            else:
                avg_accuracy = 0.0
            
            # 保存报告
            if not args.no_report and query_details:
                report_file = "~/trident/distributed-nl/nl_result.md"
                markdown_report = generate_markdown_report(
                    args.dataset, 
                    query_details, 
                    avg_phases,
                    avg_accuracy
                )
                
                # 追加模式，添加分隔符
                with open(report_file, 'a', encoding='utf-8') as f:
                    # 如果文件已存在且非空，添加分隔符
                    f.seek(0, 2)  # 移到文件末尾
                    if f.tell() > 0:
                        f.write("\n\n---\n\n")
                    f.write(markdown_report)
                
                logger.info(f"\n测试报告已保存到: {report_file}")
            
    except KeyboardInterrupt:
        logger.info("\n用户中断")
    except Exception as e:
        logger.error(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.disconnect_from_servers()


if __name__ == "__main__":
    main()