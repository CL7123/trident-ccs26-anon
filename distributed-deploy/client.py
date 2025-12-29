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
        1: {"host": "192.168.1.101", "port": 8001},
        2: {"host": "192.168.1.102", "port": 8002},
        3: {"host": "192.168.1.103", "port": 8003}
    }

# 设置日志
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [Client] %(message)s')
logger = logging.getLogger(__name__)


class DistributedClient:
    """分布式环境的测试客户端"""
    
    def __init__(self, dataset: str = "siftsmall", servers_config: Dict = None):
        self.dataset = dataset
        self.config = get_config(dataset)
        self.dpf_wrapper = OptimizedVDPFVectorWrapper(dataset_name=dataset)
        self.mpc = MPC23SSS(self.config)
        
        # 预加载原始数据用于验证
        data_dir = f"~/trident/dataset/{dataset}"
        loader = DatasetLoader(data_dir)
        self.original_nodes = loader.load_nodes()
        logger.info(f"预加载了 {len(self.original_nodes)} 个节点向量用于验证")
        logger.info(f"数据集: {dataset}, 节点数据类型: {self.original_nodes.dtype}, 形状: {self.original_nodes.shape}")
        
        # 服务器配置
        self.servers_config = servers_config or SERVERS
        self.connections = {}
        self.connection_retry_count = 3
        self.connection_timeout = 10
        
    def connect_to_servers(self):
        """连接到所有服务器，支持重试机制"""
        successful_connections = 0
        
        for server_id, server_info in self.servers_config.items():
            host = server_info["host"]
            port = server_info["port"]
            
            connected = False
            for attempt in range(self.connection_retry_count):
                try:
                    logger.info(f"尝试连接到服务器 {server_id} ({host}:{port})，第 {attempt + 1} 次...")
                    
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(self.connection_timeout)
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    
                    sock.connect((host, port))
                    self.connections[server_id] = sock
                    logger.info(f"成功连接到服务器 {server_id}")
                    successful_connections += 1
                    connected = True
                    break
                    
                except socket.timeout:
                    logger.warning(f"连接服务器 {server_id} 超时")
                except ConnectionRefusedError:
                    logger.warning(f"服务器 {server_id} 拒绝连接")
                except Exception as e:
                    logger.warning(f"连接服务器 {server_id} 失败: {e}")
                
                if attempt < self.connection_retry_count - 1:
                    time.sleep(2)  # 重试前等待
            
            if not connected:
                logger.error(f"无法连接到服务器 {server_id}")
        
        logger.info(f"成功连接到 {successful_connections}/{len(self.servers_config)} 个服务器")
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
    
    def test_distributed_query(self, node_id: int = 1723):
        """测试分布式查询"""
        # 生成VDPF密钥
        keys = self.dpf_wrapper.generate_keys('node', node_id)
        
        # 生成查询ID
        query_id = f'distributed_test_{time.time()}_{node_id}'
        
        logger.info(f"开始分布式查询，节点ID: {node_id}, 查询ID: {query_id}")
        
        # 并行查询所有服务器
        start_time = time.time()
        
        def query_server(server_id):
            request = {
                'command': 'query_node_vector',
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
        
        # 重构最终结果
        final_result = self._reconstruct_final_result(successful_responses)
        
        # 验证结果
        similarity = self._verify_result(node_id, final_result)
        
        # 打印结果
        total_time = time.time() - start_time
        logger.info(f"\n查询结果:")
        logger.info(f"  客户端总时间: {total_time:.2f}秒")
        logger.info(f"  阶段1 (VDPF评估): {avg_timings['phase1']:.2f}秒")
        logger.info(f"  阶段2 (e/f计算): {avg_timings['phase2']:.2f}秒")
        logger.info(f"  阶段3 (数据交换): {avg_timings['phase3']:.2f}秒")
        logger.info(f"  阶段4 (重构): {avg_timings['phase4']:.2f}秒")
        logger.info(f"  服务器平均总计: {avg_timings['total']:.2f}秒")
        if similarity is not None:
            logger.info(f"  余弦相似度: {similarity:.6f}")
        
        return avg_timings, final_result
    
    def _reconstruct_final_result(self, results):
        """重构最终结果"""
        # 获取至少两个服务器的响应
        server_ids = sorted([sid for sid, r in results.items() 
                           if r and r.get('status') == 'success'])[:2]
        
        if len(server_ids) < 2:
            logger.error("重构失败：可用服务器少于2个")
            return np.zeros(512 if self.dataset == "laion" else 128, dtype=np.float32)
        
        # 获取向量维度
        first_result = results[server_ids[0]]['result_share']
        vector_dim = len(first_result)
        
        # 重构每个维度
        reconstructed_vector = np.zeros(vector_dim, dtype=np.float32)
        
        # 缩放因子
        if self.dataset == "siftsmall":
            scale_factor = 1048576  # 2^20
        else:
            scale_factor = 536870912  # 2^29
        
        for i in range(vector_dim):
            shares = [
                Share(results[server_ids[0]]['result_share'][i], server_ids[0]),
                Share(results[server_ids[1]]['result_share'][i], server_ids[1])
            ]
            
            reconstructed = self.mpc.reconstruct(shares)
            
            # 转换回浮点数
            if reconstructed > self.config.prime // 2:
                signed = reconstructed - self.config.prime
            else:
                signed = reconstructed
            
            reconstructed_vector[i] = signed / scale_factor
        
        return reconstructed_vector
    
    def _verify_result(self, node_id: int, reconstructed_vector: np.ndarray):
        """验证重构结果的正确性"""
        try:
            if node_id < len(self.original_nodes):
                original_vector = self.original_nodes[node_id]
                
                # 计算余弦相似度
                dot_product = np.dot(reconstructed_vector, original_vector)
                norm_reconstructed = np.linalg.norm(reconstructed_vector)
                norm_original = np.linalg.norm(original_vector)
                
                if norm_reconstructed > 0 and norm_original > 0:
                    similarity = dot_product / (norm_reconstructed * norm_original)
                    return similarity
                else:
                    return None
            else:
                return None
                
        except Exception as e:
            logger.error(f"验证结果时出错: {e}")
            return None
    
    def get_server_status(self):
        """获取所有服务器状态"""
        logger.info("获取服务器状态...")
        
        for server_id in self.connections:
            request = {'command': 'get_status'}
            response = self._send_request(server_id, request)
            
            if response and response.get('status') == 'success':
                logger.info(f"\n服务器 {server_id} 状态:")
                logger.info(f"  模式: {response.get('mode')}")
                logger.info(f"  地址: {response.get('host')}:{response.get('port')}")
                logger.info(f"  数据集: {response.get('dataset')}")
                logger.info(f"  VDPF进程数: {response.get('vdpf_processes')}")
                logger.info(f"  数据加载: {response.get('data_loaded')}")
                logger.info(f"  可用三元组: {response.get('triples_available')}")
            else:
                logger.error(f"无法获取服务器 {server_id} 的状态")
    
    def disconnect_from_servers(self):
        """断开所有服务器连接"""
        for server_id, sock in self.connections.items():
            try:
                sock.close()
                logger.info(f"已断开与服务器 {server_id} 的连接")
            except:
                pass
        self.connections.clear()


def generate_markdown_report(dataset, query_details, avg_phases, avg_similarity):
    """生成Markdown格式的测试报告"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    markdown = f"""# 分布式测试结果报告 - {dataset}

**生成时间**: {timestamp}  
**数据集**: {dataset}  
**查询次数**: {len(query_details)}

## 详细查询结果

| 查询编号 | 节点ID | 阶段1 (VDPF) | 阶段2 (e/f) | 阶段3 (交换) | 阶段4 (重构) | 总时间 | 余弦相似度 |
|---------|--------|--------------|-------------|--------------|--------------|--------|-----------|
"""
    
    for q in query_details:
        markdown += f"| {q['query_num']} | {q['node_id']} | "
        markdown += f"{q['timings']['phase1']:.2f}s | "
        markdown += f"{q['timings']['phase2']:.2f}s | "
        markdown += f"{q['timings']['phase3']:.2f}s | "
        markdown += f"{q['timings']['phase4']:.2f}s | "
        markdown += f"{q['timings']['total']:.2f}s | "
        markdown += f"{q['similarity']:.6f} |\n"
    
    markdown += f"""
## 平均性能统计

- **阶段1 (VDPF评估)**: {avg_phases['phase1']:.2f}秒
- **阶段2 (e/f计算)**: {avg_phases['phase2']:.2f}秒
- **阶段3 (数据交换)**: {avg_phases['phase3']:.2f}秒
- **阶段4 (重构)**: {avg_phases['phase4']:.2f}秒
- **服务器平均总计**: {avg_phases['total']:.2f}秒
- **平均余弦相似度**: {avg_similarity:.6f}

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
    parser = argparse.ArgumentParser(description='分布式向量查询客户端')
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
    
    logger.info(f"=== 分布式测试客户端 - 数据集: {args.dataset} ===")
    
    client = DistributedClient(args.dataset, servers_config)
    
    try:
        # 连接服务器
        if not client.connect_to_servers():
            logger.error("连接服务器失败")
            return
        
        # 如果只是获取状态
        if args.status_only:
            client.get_server_status()
            return
        
        all_timings = []
        all_similarities = []
        query_details = []
        
        # 获取节点总数
        total_nodes = len(client.original_nodes)
        
        # 随机选择节点
        random_nodes = random.sample(range(total_nodes), min(args.num_queries, total_nodes))
        
        logger.info(f"将对 {len(random_nodes)} 个随机节点进行查询测试...\n")
        
        for i, node_id in enumerate(random_nodes):
            logger.info(f"查询 {i+1}/{len(random_nodes)}: 节点 {node_id}")
            timings, final_result = client.test_distributed_query(node_id=node_id)
            
            if timings:
                all_timings.append(timings)
                similarity = client._verify_result(node_id, final_result)
                if similarity is not None:
                    all_similarities.append(similarity)
                
                query_details.append({
                    'query_num': i + 1,
                    'node_id': node_id,
                    'timings': timings,
                    'similarity': similarity if similarity is not None else 0.0
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
            
            if all_similarities:
                avg_similarity = np.mean(all_similarities)
                logger.info(f"  平均余弦相似度: {avg_similarity:.6f}")
            else:
                avg_similarity = 0.0
            
            # 保存报告
            if not args.no_report and query_details:
                report_file = "~/trident/distributed-deploy/distributed_result.md"
                markdown_report = generate_markdown_report(
                    args.dataset, 
                    query_details, 
                    avg_phases,
                    avg_similarity
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