#!/usr/bin/env python3
"""
优化版分布式服务器 - 阶段2和阶段4使用多进程并行
"""

import sys
import os
import socket
import json
import numpy as np
import time
import hashlib
import concurrent.futures
from multiprocessing import Pool, cpu_count
import multiprocessing
import argparse
import logging
import psutil
from typing import Dict, List, Tuple, Optional, Any
import gc
import ctypes
import struct
import threading

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('~/trident/src')
sys.path.append('~/trident/standardDPF')
sys.path.append('~/trident/query-opti')

from dpf_wrapper import VDPFVectorWrapper
from dpf_wrapper_optimized import OptimizedVDPFVectorWrapper
from binary_protocol import BinaryProtocol
from basic_functionalities import get_config, MPC23SSS, Share
from secure_multiplication import NumpyMultiplicationServer

# CPU亲和性设置
try:
    sys.path.append('~/trident/query-opti')
    from cpu_affinity_optimizer import set_process_affinity
    total_cores = cpu_count()
    
    # 每个服务器有自己独立的32个物理核心
    CPU_AFFINITY_AVAILABLE = True
except Exception as e:
    print(f"CPU亲和性设置不可用: {e}")
    CPU_AFFINITY_AVAILABLE = False

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==================== 阶段2并行计算函数 ====================

def phase2_compute_batch(args):
    """阶段2的批处理计算函数"""
    batch_start, batch_end, selector_shares, data_shares, mult_triples, field_size = args
    
    batch_size = batch_end - batch_start
    vector_dim = data_shares.shape[1]
    
    # 本地计算
    all_e_shares = np.zeros(batch_size, dtype=np.uint64)
    all_f_shares = np.zeros((batch_size, vector_dim), dtype=np.uint64)
    computation_states = {}
    
    for local_idx in range(batch_size):
        global_idx = batch_start + local_idx
        
        # 获取三元组
        a, b, c = mult_triples[global_idx]
        
        # 计算e_share和f_shares
        x_value = int(selector_shares[global_idx])
        y_values = data_shares[global_idx].astype(np.uint64)
        
        e_value = (x_value - a) % field_size
        all_e_shares[local_idx] = e_value
        
        f_values = (y_values - b) % field_size
        all_f_shares[local_idx] = f_values
        
        # 保存状态
        computation_states[global_idx] = {
            'computation_id': f'query_{global_idx}',
            'triple': (a, b, c),
            'a': a,
            'b': b,
            'c': c
        }
    
    return batch_start, batch_end, all_e_shares, all_f_shares, computation_states


# ==================== 阶段4并行计算函数 ====================

def phase4_reconstruct_batch(args):
    """阶段4的批处理重构函数"""
    (batch_start, batch_end, e_shares_local, f_shares_local, 
     e_shares_others, f_shares_others, computation_states,
     lagrange_1, lagrange_2, field_size) = args
    
    batch_size = batch_end - batch_start
    vector_dim = f_shares_local.shape[1]
    
    # 准备数据矩阵
    e_shares_matrix = np.zeros((batch_size, 3), dtype=np.uint64)
    e_shares_matrix[:, 0] = e_shares_local
    
    f_shares_matrix = np.zeros((batch_size, vector_dim, 3), dtype=np.uint64)
    f_shares_matrix[:, :, 0] = f_shares_local
    
    # 填充其他服务器的数据
    for other_id, other_e in e_shares_others.items():
        e_shares_matrix[:, other_id - 1] = other_e[batch_start:batch_end]
    
    for other_id, other_f in f_shares_others.items():
        f_shares_matrix[:, :, other_id - 1] = other_f[batch_start:batch_end, :]
    
    # 拉格朗日插值重构
    e_reconstructed = (e_shares_matrix[:, 0] * lagrange_1 + 
                      e_shares_matrix[:, 1] * lagrange_2) % field_size
    
    f_reconstructed = (f_shares_matrix[:, :, 0] * lagrange_1 + 
                      f_shares_matrix[:, :, 1] * lagrange_2) % field_size
    
    # 获取三元组
    batch_a = np.zeros(batch_size, dtype=np.uint64)
    batch_b = np.zeros(batch_size, dtype=np.uint64)
    batch_c = np.zeros(batch_size, dtype=np.uint64)
    
    for local_idx in range(batch_size):
        global_idx = batch_start + local_idx
        state = computation_states[global_idx]
        batch_a[local_idx] = state['a']
        batch_b[local_idx] = state['b']
        batch_c[local_idx] = state['c']
    
    # 计算最终结果
    e_expanded = e_reconstructed[:, np.newaxis]
    a_expanded = batch_a[:, np.newaxis]
    b_expanded = batch_b[:, np.newaxis]
    c_expanded = batch_c[:, np.newaxis]
    
    result = c_expanded
    result = (result + e_expanded * b_expanded) % field_size
    result = (result + f_reconstructed * a_expanded) % field_size
    result = (result + e_expanded * f_reconstructed) % field_size
    
    # 累加贡献
    contribution = np.sum(result, axis=0) % field_size
    
    return contribution


class OptimizedDistributedServer:
    """优化的分布式服务器"""
    
    def __init__(self, server_id: int, dataset: str = "siftsmall", vdpf_processes: int = 32):
        self.server_id = server_id
        self.dataset = dataset
        self.config = get_config(dataset)
        self.field_size = self.config.prime
        self.is_query_processing = False
        
        # 多进程优化参数
        self.vdpf_processes = vdpf_processes
        self.phase2_processes = 28  # 阶段2使用28个进程
        self.phase4_processes = 28  # 阶段4使用28个进程
        self.cache_batch_size = max(100, 1000 // max(vdpf_processes // 4, 1))
        
        logger.info(f"使用配置: VDPF进程={vdpf_processes}, 阶段2进程={self.phase2_processes}, 阶段4进程={self.phase4_processes}")
        
        # 创建进程池
        self.vdpf_pool = Pool(processes=self.vdpf_processes)
        self.phase2_pool = Pool(processes=self.phase2_processes)
        self.phase4_pool = Pool(processes=self.phase4_processes)
        
        # 线程池用于I/O操作
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=8,
            thread_name_prefix=f"Server{server_id}-IO"
        )
        
        # 初始化组件
        self.dpf_wrapper = OptimizedVDPFVectorWrapper(dataset_name=dataset)
        self.mpc = MPC23SSS(self.config)
        self.mult_server = NumpyMultiplicationServer(server_id, self.config)
        
        # 加载数据
        self._load_data()
        
        # 网络设置
        self.host = '0.0.0.0'
        self.port = 8000 + server_id
        self.servers_config = self._load_servers_config()
        
        # 持久连接
        self.server_connections = {}
        self.connections_established = False
        self.connection_lock = threading.Lock()
        
        # 数据交换存储
        self.exchange_data = {}
        
    def _load_data(self):
        """加载向量级秘密共享数据"""
        logger.info(f"加载{self.dataset}数据...")
        
        self.data_dir = f"~/trident/dataset/{self.dataset}/server_{self.server_id}"
        
        # 加载节点向量份额
        self.nodes_path = os.path.join(self.data_dir, "nodes_shares.npy")
        self.node_shares = np.load(self.nodes_path)
        logger.info(f"节点数据: {self.node_shares.shape}")
        logger.info(f"数据大小: {self.node_shares.nbytes / 1024 / 1024:.1f}MB")
        
        # 预加载三元组
        logger.info(f"三元组已自动从本地目录加载")
        logger.info(f"可用三元组数量: {len(self.mult_server.triple_array) if self.mult_server.triple_array is not None else 0}")
        
    def _load_servers_config(self) -> Dict:
        """加载服务器配置"""
        try:
            from config import SERVERS
            return SERVERS
        except ImportError:
            return {
                1: {"host": "192.168.1.101", "port": 8001},
                2: {"host": "192.168.1.102", "port": 8002},
                3: {"host": "192.168.1.103", "port": 8003}
            }
    
    def process_query(self, dpf_key: bytes, query_id: str = None) -> Dict:
        """处理查询请求 - 优化版本"""
        try:
            if self.is_query_processing:
                return {'status': 'error', 'message': 'Server is busy processing another query'}
            
            self.is_query_processing = True
            start_time = time.time()
            query_id = query_id or f'query_{int(time.time()*1000)}'
            
            logger.info(f"开始处理查询 {query_id}")
            
            # 参数
            num_nodes = len(self.node_shares)
            vector_dim = self.node_shares.shape[1]
            result_accumulator = np.zeros(vector_dim, dtype=np.uint64)
            
            # 计算批次数
            num_batches = (num_nodes + self.cache_batch_size - 1) // self.cache_batch_size
            
            # ========== 阶段1：VDPF评估（保持原有的多进程） ==========
            logger.info(f"阶段1: 多进程VDPF评估 ({self.vdpf_processes} 进程)...")
            phase1_start = time.time()
            
            # 负载均衡分配
            nodes_per_process = num_nodes // self.vdpf_processes
            remaining_nodes = num_nodes % self.vdpf_processes
            
            process_args = []
            current_node_start = 0
            
            for process_id in range(self.vdpf_processes):
                process_nodes = nodes_per_process + (1 if process_id < remaining_nodes else 0)
                
                if process_nodes > 0:
                    process_args.append((
                        dpf_key,
                        process_id,
                        current_node_start,
                        current_node_start + process_nodes
                    ))
                    current_node_start += process_nodes
            
            # 设置环境变量传递 server_id
            os.environ['SERVER_ID'] = str(self.server_id)
            
            # 执行VDPF评估
            all_results = self.vdpf_pool.map(vdpf_evaluate_range_optimized, process_args)
            
            # 合并结果
            selector_values = np.zeros(num_nodes, dtype=np.uint32)
            for node_start, node_end, results in all_results:
                selector_values[node_start:node_end] = results
            
            phase1_time = time.time() - phase1_start
            logger.info(f"阶段1完成，耗时 {phase1_time:.2f}秒")
            
            # ========== 阶段2：e/f计算（多进程并行） ==========
            logger.info(f"阶段2: 多进程e/f计算 ({self.phase2_processes} 进程)...")
            phase2_start = time.time()
            
            # 准备批次参数
            phase2_args = []
            mult_triples = []
            
            # 预先获取所有三元组
            for i in range(num_nodes):
                mult_triples.append(self.mult_server.get_next_triple())
            
            # 计算每个进程的批次
            nodes_per_process = num_nodes // self.phase2_processes
            remaining = num_nodes % self.phase2_processes
            
            current_start = 0
            for i in range(self.phase2_processes):
                process_nodes = nodes_per_process + (1 if i < remaining else 0)
                if process_nodes > 0:
                    phase2_args.append((
                        current_start,
                        current_start + process_nodes,
                        selector_values,
                        self.node_shares,
                        mult_triples,
                        self.field_size
                    ))
                    current_start += process_nodes
            
            # 并行执行阶段2
            phase2_results = self.phase2_pool.map(phase2_compute_batch, phase2_args)
            
            # 合并结果
            all_e_shares = np.zeros(num_nodes, dtype=np.uint64)
            all_f_shares = np.zeros((num_nodes, vector_dim), dtype=np.uint64)
            all_computation_states = {}
            
            for batch_start, batch_end, e_shares, f_shares, states in phase2_results:
                all_e_shares[batch_start:batch_end] = e_shares
                all_f_shares[batch_start:batch_end] = f_shares
                all_computation_states.update(states)
            
            phase2_time = time.time() - phase2_start
            logger.info(f"阶段2完成，耗时 {phase2_time:.2f}秒")
            
            # ========== 阶段3：数据交换（保持原有实现） ==========
            logger.info("阶段3: 数据交换...")
            phase3_start = time.time()
            
            all_e_from_others, all_f_from_others = self._exchange_data_with_servers(
                query_id, all_e_shares, all_f_shares
            )
            
            phase3_time = time.time() - phase3_start
            logger.info(f"阶段3完成，耗时 {phase3_time:.2f}秒")
            
            # ========== 阶段4：重构计算（多进程并行） ==========
            logger.info(f"阶段4: 多进程重构计算 ({self.phase4_processes} 进程)...")
            phase4_start = time.time()
            
            # 拉格朗日系数
            lagrange_1 = 2
            lagrange_2 = self.field_size - 1
            
            # 准备批次参数
            phase4_args = []
            nodes_per_process = num_nodes // self.phase4_processes
            remaining = num_nodes % self.phase4_processes
            
            current_start = 0
            for i in range(self.phase4_processes):
                process_nodes = nodes_per_process + (1 if i < remaining else 0)
                if process_nodes > 0:
                    phase4_args.append((
                        current_start,
                        current_start + process_nodes,
                        all_e_shares[current_start:current_start + process_nodes],
                        all_f_shares[current_start:current_start + process_nodes],
                        all_e_from_others,
                        all_f_from_others,
                        all_computation_states,
                        lagrange_1,
                        lagrange_2,
                        self.field_size
                    ))
                    current_start += process_nodes
            
            # 并行执行阶段4
            phase4_results = self.phase4_pool.map(phase4_reconstruct_batch, phase4_args)
            
            # 累加所有贡献
            for contribution in phase4_results:
                result_accumulator = (result_accumulator + contribution) % self.field_size
            
            phase4_time = time.time() - phase4_start
            total_time = time.time() - start_time
            
            logger.info(f"查询完成:")
            logger.info(f"  阶段1 (VDPF): {phase1_time:.2f}秒")
            logger.info(f"  阶段2 (e/f): {phase2_time:.2f}秒")
            logger.info(f"  阶段3 (交换): {phase3_time:.2f}秒")
            logger.info(f"  阶段4 (重构): {phase4_time:.2f}秒")
            logger.info(f"  总计: {total_time:.2f}秒")
            
            # 清理
            self._cleanup_query_files(query_id)
            
            # 返回结果
            result_list = [int(x) % (2**32) for x in result_accumulator]
            
            response = {
                'status': 'success',
                'server_id': self.server_id,
                'result_share': result_list,
                'timing': {
                    'phase1_time': phase1_time * 1000,
                    'phase2_time': phase2_time * 1000,
                    'phase3_time': phase3_time * 1000,
                    'phase4_time': phase4_time * 1000,
                    'total': total_time * 1000,
                    'triples_used': num_nodes
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"查询处理错误: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}
        finally:
            self.is_query_processing = False
    
    def _exchange_data_with_servers(self, query_id: str, e_shares: np.ndarray, f_shares: np.ndarray):
        """与其他服务器交换数据（保持原有实现）"""
        all_e_from_others = {}
        all_f_from_others = {}
        
        # 使用线程池并行发送
        def send_to_server_async(target_id):
            success = self._send_binary_exchange_data(target_id, query_id, e_shares, f_shares)
            return target_id, success
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(send_to_server_async, sid) for sid in [1, 2, 3] if sid != self.server_id]
            concurrent.futures.wait(futures)
            
            for future in futures:
                target_id, success = future.result()
                if success:
                    logger.info(f"成功发送数据到服务器 {target_id}")
                else:
                    logger.error(f"发送数据到服务器 {target_id} 失败")
        
        # 接收其他服务器的数据
        all_e_from_others, all_f_from_others = self._receive_all_exchange_data(query_id)
        
        return all_e_from_others, all_f_from_others
    
    def _send_binary_exchange_data(self, target_server_id: int, query_id: str, 
                                   e_shares: np.ndarray, f_shares: np.ndarray) -> bool:
        """发送二进制格式的交换数据"""
        # 第一次调用时建立所有连接
        if not self.connections_established:
            logger.info("第一次数据交换，建立持久连接...")
            self._establish_persistent_connections()
            self.connections_established = True
        
        # 检查连接
        if target_server_id not in self.server_connections:
            logger.warning(f"没有到服务器 {target_server_id} 的持久连接")
            return False
        
        conn = self.server_connections[target_server_id]
        
        try:
            # 发送命令
            command = {
                'command': 'exchange_data',
                'query_id': query_id,
                'from_server': self.server_id
            }
            command_data = json.dumps(command).encode()
            conn.sendall(len(command_data).to_bytes(4, 'big'))
            conn.sendall(command_data)
            
            # 准备二进制数据
            vector_dim = f_shares.shape[1]
            header = struct.pack('!III', len(e_shares), vector_dim, self.server_id)
            e_bytes = e_shares.astype(np.float32).tobytes()
            f_bytes = f_shares.astype(np.float32).tobytes()
            
            data = header + e_bytes + f_bytes
            
            # 发送数据
            conn.sendall(len(data).to_bytes(4, 'big'))
            conn.sendall(data)
            
            # 接收确认
            ack_length_bytes = conn.recv(4)
            if ack_length_bytes:
                ack_length = int.from_bytes(ack_length_bytes, 'big')
                ack_data = conn.recv(ack_length)
                ack = json.loads(ack_data.decode())
                return ack.get('status') == 'received'
            
            return False
            
        except Exception as e:
            logger.error(f"发送数据到服务器 {target_server_id} 时出错: {e}")
            return False
    
    def _receive_all_exchange_data(self, query_id: str):
        """接收所有其他服务器的交换数据"""
        all_e_from_others = {}
        all_f_from_others = {}
        
        # 等待接收数据
        max_wait_time = 30
        start_wait = time.time()
        expected_servers = [sid for sid in [1, 2, 3] if sid != self.server_id]
        
        while len(all_e_from_others) < len(expected_servers):
            if time.time() - start_wait > max_wait_time:
                logger.error(f"等待交换数据超时")
                break
            
            with self.connection_lock:
                for from_server in expected_servers:
                    if from_server not in all_e_from_others:
                        key = f"{query_id}_from_{from_server}"
                        if key in self.exchange_data:
                            data = self.exchange_data[key]
                            all_e_from_others[from_server] = data['e_shares']
                            all_f_from_others[from_server] = data['f_shares']
                            logger.info(f"获取到服务器 {from_server} 的交换数据")
                            del self.exchange_data[key]
            
            if len(all_e_from_others) < len(expected_servers):
                time.sleep(0.01)
        
        logger.info(f"收到 {len(all_e_from_others)}/{len(expected_servers)} 个服务器的数据")
        
        return all_e_from_others, all_f_from_others
    
    def _establish_persistent_connections(self):
        """建立到其他服务器的持久连接"""
        for server_id, server_info in self.servers_config.items():
            if server_id != self.server_id:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                    sock.connect((server_info['host'], server_info['port']))
                    
                    self.server_connections[server_id] = sock
                    logger.info(f"建立到服务器 {server_id} 的持久连接")
                    
                except Exception as e:
                    logger.error(f"无法连接到服务器 {server_id}: {e}")
    
    def _cleanup_query_files(self, query_id: str):
        """清理查询相关文件"""
        pass
    
    def _handle_client(self, client_socket, client_address):
        """处理客户端连接"""
        logger.info(f"新客户端连接: {client_address}")
        
        try:
            while True:
                # 接收请求长度
                length_bytes = client_socket.recv(4)
                if not length_bytes:
                    break
                
                length = int.from_bytes(length_bytes, 'big')
                
                # 接收请求数据
                data = b''
                while len(data) < length:
                    chunk = client_socket.recv(min(length - len(data), 4096))
                    if not chunk:
                        break
                    data += chunk
                
                if len(data) < length:
                    break
                
                # 解析请求
                try:
                    # 检查是否是二进制协议（通过第一个字节判断）
                    if len(data) > 0 and data[0] in [BinaryProtocol.CMD_QUERY_NODE_VECTOR, BinaryProtocol.CMD_GET_STATUS]:
                        request = BinaryProtocol.decode_request(data)
                        logger.info(f"收到二进制请求: {request.get('command', 'unknown')}")
                        
                        if request.get('command') == 'query_node_vector':
                            dpf_key = request.get('dpf_key')
                            query_id = request.get('query_id')
                            response = self.process_query(dpf_key, query_id)
                            response_data = BinaryProtocol.encode_response(response)
                            client_socket.sendall(response_data)
                    else:
                        # JSON请求
                        request = json.loads(data.decode())
                        logger.info(f"收到JSON请求: {request.get('command', 'unknown')}")
                        
                        # 特殊处理数据交换请求
                        if request.get('command') == 'exchange_data':
                            self._handle_exchange_data(request, client_socket)
                        else:
                            response = self._handle_request(request)
                            response_data = json.dumps(response).encode()
                            client_socket.sendall(len(response_data).to_bytes(4, 'big'))
                            client_socket.sendall(response_data)
                            
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析失败: {e}")
                    error_response = {'status': 'error', 'message': 'Invalid JSON format'}
                    error_data = json.dumps(error_response).encode()
                    client_socket.sendall(len(error_data).to_bytes(4, 'big'))
                    client_socket.sendall(error_data)
                except Exception as e:
                    logger.error(f"解析请求失败: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"处理客户端请求时出错: {e}")
        finally:
            client_socket.close()
            logger.info(f"客户端连接关闭: {client_address}")
    
    def _handle_request(self, request: Dict) -> Dict:
        """处理非查询请求"""
        command = request.get('command')
        
        if command == 'get_status':
            return {
                'status': 'success',
                'server_id': self.server_id,
                'mode': 'distributed',
                'host': self.host,
                'port': self.port,
                'dataset': self.dataset,
                'vdpf_processes': self.vdpf_processes,
                'phase2_processes': self.phase2_processes,
                'phase4_processes': self.phase4_processes,
                'data_loaded': {
                    'nodes': self.node_shares.shape
                },
                'triples_available': len(self.mult_server.triple_array) if self.mult_server.triple_array is not None else 0
            }
        
        elif command == 'exchange_data':
            # 处理数据交换请求
            return self._handle_exchange_data(request)
        
        else:
            return {'status': 'error', 'message': f'Unknown command: {command}'}
    
    def _handle_exchange_data(self, request: Dict, client_socket) -> None:
        """处理数据交换请求"""
        query_id = request.get('query_id')
        from_server = request.get('from_server')
        
        try:
            # 接收二进制数据长度
            length_bytes = client_socket.recv(4)
            if not length_bytes:
                return
            
            length = int.from_bytes(length_bytes, 'big')
            
            # 接收二进制数据
            data = b''
            while len(data) < length:
                chunk = client_socket.recv(min(length - len(data), 65536))
                if not chunk:
                    break
                data += chunk
            
            # 解析二进制数据
            header_size = 12  # 3 * 4 bytes for III format
            num_nodes, vector_dim, server_id = struct.unpack('!III', data[:header_size])
            
            # 提取数据
            offset = header_size
            e_size = num_nodes * 4  # float32
            f_size = num_nodes * vector_dim * 4  # float32
            
            e_bytes = data[offset:offset + e_size]
            f_bytes = data[offset + e_size:offset + e_size + f_size]
            
            # 转换为numpy数组
            e_shares = np.frombuffer(e_bytes, dtype=np.float32).astype(np.uint64)
            f_shares = np.frombuffer(f_bytes, dtype=np.float32).reshape(num_nodes, vector_dim).astype(np.uint64)
            
            # 存储数据
            key = f"{query_id}_from_{from_server}"
            with self.connection_lock:
                self.exchange_data[key] = {
                    'e_shares': e_shares,
                    'f_shares': f_shares
                }
            
            logger.info(f"接收到服务器 {from_server} 的交换数据: e_shares形状={e_shares.shape}, f_shares形状={f_shares.shape}")
            
            # 发送确认
            ack = {'status': 'received'}
            ack_data = json.dumps(ack).encode()
            client_socket.sendall(len(ack_data).to_bytes(4, 'big'))
            client_socket.sendall(ack_data)
            
        except Exception as e:
            logger.error(f"处理交换数据时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def start(self):
        """启动服务器"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)
        
        logger.info(f"优化服务器 {self.server_id} 启动")
        logger.info(f"监听地址: {self.host}:{self.port}")
        logger.info(f"数据集: {self.dataset}")
        logger.info(f"VDPF进程数: {self.vdpf_processes}")
        logger.info(f"阶段2进程数: {self.phase2_processes}")
        logger.info(f"阶段4进程数: {self.phase4_processes}")
        
        # 预热进程池
        logger.info("预热进程池...")
        warmup_start = time.time()
        
        # 预热所有进程池
        self.vdpf_pool.map(warmup_process, range(self.vdpf_processes))
        self.phase2_pool.map(warmup_process, range(self.phase2_processes))
        self.phase4_pool.map(warmup_process, range(self.phase4_processes))
        
        warmup_time = time.time() - warmup_start
        logger.info(f"进程池预热完成，耗时 {warmup_time:.2f}秒")
        
        logger.info("服务器就绪，等待连接...")
        
        try:
            while True:
                client_socket, client_address = server_socket.accept()
                # 使用线程池处理客户端
                self.executor.submit(self._handle_client, client_socket, client_address)
                
        except KeyboardInterrupt:
            logger.info("服务器关闭中...")
        finally:
            server_socket.close()
            self.vdpf_pool.close()
            self.phase2_pool.close()
            self.phase4_pool.close()
            self.executor.shutdown()


# 从原文件复制必要的辅助函数
def warmup_process(process_id):
    """预热进程，加载必要的模块"""
    import time
    import sys
    sys.path.append('~/trident/query-opti')
    from dpf_wrapper_optimized import OptimizedVDPFVectorWrapper
    
    # 简单的计算任务来预热
    _ = sum(i * i for i in range(1000))
    return process_id


def vdpf_evaluate_range_optimized(args):
    """优化的VDPF评估函数"""
    dpf_key, process_id, node_start, node_end = args
    
    # CPU亲和性设置（如果需要的话，可以在这里添加）
    # 暂时跳过，因为需要正确的函数签名
    
    from dpf_wrapper_optimized import OptimizedVDPFVectorWrapper
    from binary_serializer import BinaryKeySerializer
    
    dataset_name = os.environ.get('DATASET_NAME', 'siftsmall')
    wrapper = OptimizedVDPFVectorWrapper(dataset_name=dataset_name)
    
    # 反序列化密钥
    if isinstance(dpf_key, bytes):
        key = BinaryKeySerializer.deserialize_vdpf23_key(dpf_key)
    else:
        key = wrapper._deserialize_key(dpf_key)
    
    # 获取实际的 server_id
    server_id = int(os.environ.get('SERVER_ID', '1'))
    
    # 使用 eval_batch 方法
    batch_results = wrapper.eval_batch(key, node_start, node_end, server_id)
    
    # 提取结果
    results = []
    for node_id in range(node_start, node_end):
        if node_id in batch_results:
            results.append(batch_results[node_id])
        else:
            logger.error(f"Process {process_id}: Missing result for node {node_id}")
            results.append(0)
    
    return node_start, node_end, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='优化的分布式服务器')
    parser.add_argument('--server-id', type=int, required=True, help='服务器ID (1, 2, 或 3)')
    parser.add_argument('--dataset', type=str, default='siftsmall', help='数据集名称')
    parser.add_argument('--vdpf-processes', type=int, default=32, help='VDPF评估进程数')
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ['DATASET_NAME'] = args.dataset
    
    server = OptimizedDistributedServer(args.server_id, args.dataset, args.vdpf_processes)
    server.start()