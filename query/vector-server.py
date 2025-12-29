#!/usr/bin/env python3

import sys
import os
import socket
import json
import numpy as np
import time
import threading
from typing import Dict, List, Tuple
import pickle
import hashlib
import concurrent.futures
from multiprocessing import Pool, cpu_count
import multiprocessing
import mmap
import tempfile
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('~/trident/standardDPF')
sys.path.append('~/trident/src')

from dpf_wrapper import VDPFVectorWrapper
from secure_multiplication import NumpyMultiplicationServer
from basic_functionalities import get_config, Share, MPC23SSS


# 全局函数，用于进程池调用（必须在模块顶层定义）
def evaluate_batch_range_process(args):
    """进程池工作函数：评估指定批次范围的VDPF"""
    process_id, start_batch, end_batch, cache_batch_size, num_nodes, serialized_key, server_id, node_shares_path, dataset_name = args
    
    print(f"[Process {process_id}] 开始评估批次 {start_batch}-{end_batch-1}")
    
    # 每个进程创建独立的VDPF实例
    from dpf_wrapper import VDPFVectorWrapper
    dpf_wrapper = VDPFVectorWrapper(dataset_name=dataset_name)
    
    # 反序列化密钥
    key = dpf_wrapper._deserialize_key(serialized_key)
    
    # 加载节点数据（每个进程独立加载）
    node_shares = np.load(node_shares_path)
    
    local_selector_shares = {}
    local_vector_shares = {}
    process_start_time = time.time()
    
    for batch_idx in range(start_batch, end_batch):
        batch_start = batch_idx * cache_batch_size
        batch_end = min(batch_start + cache_batch_size, num_nodes)
        batch_size = batch_end - batch_start
        
        # 预加载整个批次到缓存中
        batch_data = node_shares[batch_start:batch_end].copy()
        
        # 顺序处理这个批次中的所有节点
        for local_idx in range(batch_size):
            global_idx = batch_start + local_idx
            
            # VDPF评估
            selector_share = dpf_wrapper.eval_at_position(key, global_idx, server_id)
            local_selector_shares[global_idx] = selector_share
            
            # 使用预加载的数据
            local_vector_shares[global_idx] = batch_data[local_idx]
    
    process_time = time.time() - process_start_time
    evaluated_nodes = sum(min(cache_batch_size, num_nodes - b * cache_batch_size) 
                        for b in range(start_batch, end_batch))
    ops_per_sec = evaluated_nodes / process_time if process_time > 0 else 0
    
    print(f"[Process {process_id}] 完成: {process_time:.3f}秒, "
          f"{evaluated_nodes} 节点, {ops_per_sec:.0f} ops/sec")
    
    return {
        'selector_shares': local_selector_shares,
        'vector_shares': local_vector_shares
    }


class MemoryMappedOptimizedServer:
    """多进程数据局部性优化的服务器（支持可配置进程数）"""
    
    def __init__(self, server_id: int, dataset: str = "laion", vdpf_processes: int = 4):
        self.server_id = server_id
        self.dataset = dataset
        self.config = get_config(dataset)
        self.field_size = self.config.prime
        self.mpc = MPC23SSS(self.config)
        
        # 网络配置
        self.host = f"192.168.50.2{server_id}"
        self.port = 8000 + server_id
        
        # 初始化组件
        self.dpf_wrapper = VDPFVectorWrapper(dataset_name=dataset)
        self.mult_server = NumpyMultiplicationServer(server_id, self.config)
        
        # 加载数据
        self._load_data()
        
        # 模拟交换的目录（回退到标准文件系统）
        self.exchange_dir = "/tmp/mpc_exchange"
        os.makedirs(self.exchange_dir, exist_ok=True)
        
        # ===== 多进程优化参数 =====
        self.cache_batch_size = 2000  # 缓存友好的批次大小
        self.vdpf_processes = vdpf_processes  # 可配置的VDPF评估进程数
        self.worker_threads = 4  # 其他操作的线程数
        
        # 为其他操作创建线程池
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.worker_threads,
            thread_name_prefix=f"Server{server_id}-General"
        )
        
        print(f"[Server {self.server_id}] 多进程数据局部性优化模式")
        print(f"[Server {self.server_id}] 缓存批次大小: {self.cache_batch_size}")
        print(f"[Server {self.server_id}] VDPF评估进程数: {self.vdpf_processes}")
        print(f"[Server {self.server_id}] 交换目录: {self.exchange_dir}")
        
    def _load_data(self):
        """加载向量级秘密共享数据"""
        print(f"[Server {self.server_id}] 加载{self.dataset}数据...")
        
        # 统一使用Test-Trident路径
        self.data_dir = f"~/trident/dataset/{self.dataset}/server_{self.server_id}"
        
        # 加载节点向量份额
        self.nodes_path = os.path.join(self.data_dir, "nodes_shares.npy")
        self.node_shares = np.load(self.nodes_path)
        print(f"  节点数据: {self.node_shares.shape}")
        print(f"  数据大小: {self.node_shares.nbytes / 1024 / 1024:.1f}MB")
        
        print(f"  三元组可用: {self.mult_server.triple_array.shape[0] - self.mult_server.used_count if self.mult_server.triple_array is not None else 0}")
        
    def start(self):
        """启动服务器"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)
        
        print(f"[Server {self.server_id}] 监听 {self.host}:{self.port}")
        
        # 清理旧的交换文件
        self._cleanup_exchange_files()
        
        try:
            while True:
                client_socket, address = server_socket.accept()
                print(f"[Server {self.server_id}] 接受连接来自 {address}")
                
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket,)
                )
                client_thread.start()
                
        except KeyboardInterrupt:
            print(f"\n[Server {self.server_id}] 关闭服务器...")
        finally:
            server_socket.close()
            self._cleanup_exchange_files()
            self.executor.shutdown(wait=True)
            print(f"[Server {self.server_id}] 服务器已关闭")
    
    def _cleanup_exchange_files(self):
        """清理交换文件"""
        pattern = f"server_{self.server_id}_"
        current_time = time.time()
        for filename in os.listdir(self.exchange_dir):
            if filename.startswith(pattern):
                filepath = os.path.join(self.exchange_dir, filename)
                try:
                    # 清理超过5分钟的旧文件
                    if os.path.getmtime(filepath) < current_time - 300:
                        os.remove(filepath)
                except:
                    pass
    
    def _handle_client(self, client_socket: socket.socket):
        """处理客户端请求"""
        try:
            while True:
                length_bytes = client_socket.recv(4)
                if not length_bytes:
                    break
                
                length = int.from_bytes(length_bytes, 'big')
                data = b''
                while len(data) < length:
                    chunk = client_socket.recv(min(length - len(data), 4096))
                    if not chunk:
                        break
                    data += chunk
                
                request = json.loads(data.decode())
                print(f"[Server {self.server_id}] 收到请求: {request.get('command', 'unknown')}")
                
                response = self._process_request(request)
                print(f"[Server {self.server_id}] 请求处理完成，响应状态: {response.get('status', 'unknown')}")
                
                response_data = json.dumps(response).encode()
                client_socket.sendall(len(response_data).to_bytes(4, 'big'))
                client_socket.sendall(response_data)
                
        except Exception as e:
            print(f"[Server {self.server_id}] 处理客户端错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            client_socket.close()
    
    def _process_request(self, request: Dict) -> Dict:
        """处理请求"""
        command = request.get('command')
        
        if command == 'query_node_vector':
            return self._handle_vector_node_query(request)
        elif command == 'get_status':
            return self._get_status()
        else:
            return {'status': 'error', 'message': f'未知命令: {command}'}
    
    def _multiprocess_vdpf_evaluation(self, serialized_key, num_nodes, num_batches):
        """多进程VDPF评估 - 核心优化函数（保持原有实现）"""
        
        # 将批次平均分配给进程
        batches_per_process = max(1, num_batches // self.vdpf_processes)
        remaining_batches = num_batches % self.vdpf_processes
        
        # 准备进程参数
        process_args = []
        current_batch = 0
        
        for process_id in range(self.vdpf_processes):
            start_batch = current_batch
            process_batches = batches_per_process + (1 if process_id < remaining_batches else 0)
            end_batch = start_batch + process_batches
            
            if start_batch < num_batches:
                args = (
                    process_id,
                    start_batch,
                    min(end_batch, num_batches),
                    self.cache_batch_size,
                    num_nodes,
                    serialized_key,
                    self.server_id,
                    self.nodes_path,  # 传递文件路径而不是数据
                    self.dataset  # 传递数据集名称
                )
                process_args.append(args)
                current_batch = end_batch
        
        # 使用进程池并行执行
        print(f"[Server {self.server_id}] 启动 {len(process_args)} 个进程进行VDPF评估")
        
        with Pool(processes=self.vdpf_processes) as pool:
            results = pool.map(evaluate_batch_range_process, process_args)
        
        # 合并所有进程的结果
        all_selector_shares = {}
        all_vector_shares = {}
        
        for process_result in results:
            all_selector_shares.update(process_result['selector_shares'])
            all_vector_shares.update(process_result['vector_shares'])
        
        print(f"[Server {self.server_id}] 多进程VDPF评估完成，共评估 {len(all_selector_shares)} 个节点")
        
        return all_selector_shares, all_vector_shares
    
    def _save_exchange_data(self, query_id: str, e_shares: np.ndarray, f_shares: np.ndarray):
        """保存交换数据到文件（无压缩优化）"""
        
        # 使用.npz格式但不压缩，加快I/O速度
        filename = f"server_{self.server_id}_query_{query_id}_data.npz"
        filepath = os.path.join(self.exchange_dir, filename)
        
        # 保留校验和计算（用于数据完整性验证）
        data_hash = hashlib.md5((e_shares.tobytes() + f_shares.tobytes())).hexdigest()
        
        np.savez(filepath, 
                 e_shares=e_shares, 
                 f_shares=f_shares,
                 hash=data_hash)
        
        print(f"[Server {self.server_id}] 已保存交换数据到 {filename}（无压缩优化）")
    
    def _load_other_servers_data(self, query_id: str, num_nodes: int) -> Tuple[Dict, Dict]:
        """加载其他服务器的数据"""
        
        all_e_from_others = {}
        all_f_from_others = {}
        
        for other_id in [1, 2, 3]:
            if other_id == self.server_id:
                continue
            
            filename = f"server_{other_id}_query_{query_id}_data.npz"
            filepath = os.path.join(self.exchange_dir, filename)
            
            # 等待文件出现（保持原有等待机制）
            max_wait = 30
            for i in range(max_wait):
                if os.path.exists(filepath):
                    break
                time.sleep(1)
            
            if os.path.exists(filepath):
                # 加载无压缩的.npz文件（格式相同，只是无压缩）
                data = np.load(filepath)
                all_e_from_others[other_id] = data['e_shares']
                all_f_from_others[other_id] = data['f_shares']
                print(f"[Server {self.server_id}] 已加载 Server {other_id} 的数据（无压缩优化）")
                
                # 可选：验证数据完整性
                if 'hash' in data:
                    loaded_hash = str(data['hash'])
                    expected_hash = hashlib.md5((data['e_shares'].tobytes() + data['f_shares'].tobytes())).hexdigest()
                    if loaded_hash != expected_hash:
                        print(f"[Server {self.server_id}] 警告: Server {other_id} 数据校验和不匹配")
            else:
                print(f"[Server {self.server_id}] 警告: 无法找到 Server {other_id} 的数据")
        
        return all_e_from_others, all_f_from_others
    
    def _handle_vector_node_query(self, request: Dict) -> Dict:
        """向量级节点查询 - 多进程数据局部性优化版本"""
        try:
            serialized_key = request['dpf_key']
            query_id = request.get('query_id', 'unknown')
            
            print(f"[Server {self.server_id}] 处理向量级节点查询（多进程数据局部性优化）")
            
            # 1. 反序列化密钥（主进程中进行，验证密钥格式）
            key = self.dpf_wrapper._deserialize_key(serialized_key)
            
            # 2. 初始化
            start_time = time.time()
            num_nodes = len(self.node_shares)
            vector_dim = self.node_shares.shape[1]  # 动态获取向量维度
            result_accumulator = np.zeros(vector_dim, dtype=np.int64)
            
            print(f"[Server {self.server_id}] 节点总数: {num_nodes}, 缓存批次大小: {self.cache_batch_size}")
            print(f"[Server {self.server_id}] DEBUG: 向量维度: {vector_dim}, 查询ID: {query_id}")
            
            # 3. 阶段1：多进程VDPF评估（保持原有实现)
            print(f"[Server {self.server_id}] 阶段1: 多进程VDPF评估 ({self.vdpf_processes} 进程)...")
            phase1_start = time.time()
            
            num_batches = (num_nodes + self.cache_batch_size - 1) // self.cache_batch_size
            print(f"[Server {self.server_id}] 将处理 {num_batches} 个批次，使用 {self.vdpf_processes} 个进程")
            
            # 使用多进程评估VDPF
            all_selector_shares, all_vector_shares = self._multiprocess_vdpf_evaluation(
                serialized_key, num_nodes, num_batches)
            
            phase1_time = time.time() - phase1_start
            total_ops_per_sec = num_nodes / phase1_time if phase1_time > 0 else 0
            print(f"[Server {self.server_id}] 阶段1完成（多进程），耗时 {phase1_time:.2f}秒, 平均 {total_ops_per_sec:.0f} ops/sec")
            
            # DEBUG: 检查VDPF选择器的值
            non_zero_count = sum(1 for v in all_selector_shares.values() if v != 0)
            print(f"[Server {self.server_id}] DEBUG: VDPF非零位置数: {non_zero_count}")
            if non_zero_count > 0:
                for idx, val in list(all_selector_shares.items())[:5]:  # 打印前5个非零值
                    if val != 0:
                        print(f"[Server {self.server_id}] DEBUG: 位置 {idx} 的VDPF值: {val}")
            
            # 3.5. 文件同步屏障
            print(f"[Server {self.server_id}] 使用文件系统同步...")
            self._file_sync_barrier(query_id, "phase1")
            
            # 4. 阶段2：数据局部性优化的e/f计算（保持原有实现）
            print(f"[Server {self.server_id}] 阶段2: 缓存友好的e/f计算...")
            phase2_start = time.time()
            
            all_e_shares = np.zeros(num_nodes, dtype=np.uint64)
            all_f_shares = np.zeros((num_nodes, vector_dim), dtype=np.uint64)
            all_computation_states = {}
            
            # 同样使用批次处理，避免随机内存访问
            for batch_idx in range(num_batches):
                batch_start = batch_idx * self.cache_batch_size
                batch_end = min(batch_start + self.cache_batch_size, num_nodes)
                batch_size = batch_end - batch_start
                
                print(f"[Server {self.server_id}] e/f计算批次 {batch_idx+1}/{num_batches}")
                
                # 批量获取三元组（减少锁竞争）
                batch_triples = []
                for _ in range(batch_size):
                    batch_triples.append(self.mult_server.get_next_triple())
                
                # 顺序处理这个批次
                for local_idx in range(batch_size):
                    global_idx = batch_start + local_idx
                    computation_id = f"query_{query_id}_pos{global_idx}"
                    
                    a, b, c = batch_triples[local_idx]
                    
                    e_share = (all_selector_shares[global_idx] - a) % self.field_size
                    all_e_shares[global_idx] = e_share
                    
                    # 向量化计算所有维度的f值
                    vector_data = all_vector_shares[global_idx]
                    f_values = (vector_data.astype(np.int64) - b) % self.field_size
                    all_f_shares[global_idx] = f_values
                    
                    all_computation_states[global_idx] = {
                        'selector_share': all_selector_shares[global_idx],
                        'vector_share': all_vector_shares[global_idx],
                        'computation_id': computation_id,
                        'triple': (a, b, c)
                    }
                    
                    self.mult_server.computation_cache[computation_id] = {
                        'a': a,
                        'b': b,
                        'c': c
                    }
            
            phase2_time = time.time() - phase2_start
            print(f"[Server {self.server_id}] 阶段2完成，耗时 {phase2_time:.2f}秒")
            
            # 5. 阶段3：模拟批量交换（保持无压缩优化）
            print(f"[Server {self.server_id}] 阶段3: 模拟批量交换...")
            phase3_start = time.time()
            
            # 保存本服务器的数据
            self._save_exchange_data(query_id, all_e_shares, all_f_shares)
            
            # 等待其他服务器保存完成
            self._file_sync_barrier(query_id, "phase3_save")
            
            # 读取其他服务器的数据
            all_e_from_others, all_f_from_others = self._load_other_servers_data(query_id, num_nodes)
            
            phase3_time = time.time() - phase3_start
            print(f"[Server {self.server_id}] 阶段3完成（模拟），耗时 {phase3_time:.2f}秒")
            
            # 6. 阶段4：数据局部性优化的重构计算（保持NumPy并行化优化）
            print(f"[Server {self.server_id}] 阶段4: 缓存友好的重构计算...")
            phase4_start = time.time()
            
            # 预计算拉格朗日系数，用于所有批次
            lagrange_1 = 2
            lagrange_2 = self.field_size - 1
            
            # 使用批次处理进行并行化重构计算
            for batch_idx in range(num_batches):
                batch_start = batch_idx * self.cache_batch_size
                batch_end = min(batch_start + self.cache_batch_size, num_nodes)
                batch_size = batch_end - batch_start
                
                print(f"[Server {self.server_id}] NumPy并行化重构批次 {batch_idx+1}/{num_batches} ({batch_size} 节点)")
                
                # 1. 批量提取数据 - 避免逐个节点访问
                batch_e_shares_local = all_e_shares[batch_start:batch_end]  # shape: (batch_size,)
                batch_f_shares_local = all_f_shares[batch_start:batch_end, :]  # shape: (batch_size, vector_dim)
                
                # 2. 构建三个服务器的份额矩阵
                # e_shares_matrix: shape (batch_size, 3)
                e_shares_matrix = np.zeros((batch_size, 3), dtype=np.uint64)
                e_shares_matrix[:, self.server_id - 1] = batch_e_shares_local
                
                # f_shares_matrix: shape (batch_size, vector_dim, 3)
                f_shares_matrix = np.zeros((batch_size, vector_dim, 3), dtype=np.uint64)
                f_shares_matrix[:, :, self.server_id - 1] = batch_f_shares_local
                
                # 3. 填入其他服务器的数据
                for other_id, other_e_shares in all_e_from_others.items():
                    e_shares_matrix[:, other_id - 1] = other_e_shares[batch_start:batch_end]
                
                for other_id, other_f_shares in all_f_from_others.items():
                    f_shares_matrix[:, :, other_id - 1] = other_f_shares[batch_start:batch_end, :]
                
                # 4. 向量化重构e值 - 批量处理整个批次
                # 使用拉格朗日插值: e = e1 * 2 + e2 * (-1)
                batch_e_reconstructed = (e_shares_matrix[:, 0] * lagrange_1 + 
                                       e_shares_matrix[:, 1] * lagrange_2) % self.field_size
                # shape: (batch_size,)
                
                # 5. 向量化重构f值 - 批量处理所有维度
                # f_reconstructed: shape (batch_size, vector_dim)
                batch_f_reconstructed = (f_shares_matrix[:, :, 0] * lagrange_1 + 
                                       f_shares_matrix[:, :, 1] * lagrange_2) % self.field_size
                
                # 6. 批量获取三元组数据
                batch_triples = []
                for local_idx in range(batch_size):
                    global_idx = batch_start + local_idx
                    state = all_computation_states[global_idx]
                    a, b, c = state['triple']
                    batch_triples.append((a, b, c))
                
                # 转换为NumPy数组以便向量化
                batch_a = np.array([t[0] for t in batch_triples], dtype=np.uint64)  # shape: (batch_size,)
                batch_b = np.array([t[1] for t in batch_triples], dtype=np.uint64)  # shape: (batch_size,)
                batch_c = np.array([t[2] for t in batch_triples], dtype=np.uint64)  # shape: (batch_size,)
                
                # 7. 向量化计算最终结果 - 批量处理所有节点和维度
                # 使用广播机制计算 batch_size × vector_dim 的结果矩阵
                
                # 扩展维度以支持广播: (batch_size, 1) 广播到 (batch_size, vector_dim)
                batch_e_expanded = batch_e_reconstructed[:, np.newaxis]  # shape: (batch_size, 1)
                batch_a_expanded = batch_a[:, np.newaxis]  # shape: (batch_size, 1)
                batch_b_expanded = batch_b[:, np.newaxis]  # shape: (batch_size, 1)
                batch_c_expanded = batch_c[:, np.newaxis]  # shape: (batch_size, 1)
                
                # 计算结果: result = c + e*b + f*a + e*f (mod field_size)
                batch_result = batch_c_expanded  # shape: (batch_size, vector_dim)
                batch_result = (batch_result + batch_e_expanded * batch_b_expanded) % self.field_size
                batch_result = (batch_result + batch_f_reconstructed * batch_a_expanded) % self.field_size
                batch_result = (batch_result + batch_e_expanded * batch_f_reconstructed) % self.field_size
                
                # 8. 累加到总结果 - 对所有节点的向量维度结果求和
                batch_contribution = np.sum(batch_result, axis=0) % self.field_size  # shape: (vector_dim,)
                result_accumulator = (result_accumulator + batch_contribution) % self.field_size
                
                # DEBUG: 第一个批次的调试信息
                if batch_idx == 0:
                    print(f"[Server {self.server_id}] DEBUG: 批次1贡献的前5个值: {batch_contribution[:5]}")
                    print(f"[Server {self.server_id}] DEBUG: 累加器的前5个值: {result_accumulator[:5]}")
                
                # 9. 清理计算缓存
                for local_idx in range(batch_size):
                    global_idx = batch_start + local_idx
                    state = all_computation_states[global_idx]
                    computation_id = state['computation_id']
                    if computation_id in self.mult_server.computation_cache:
                        del self.mult_server.computation_cache[computation_id]
                
                print(f"[Server {self.server_id}] 批次 {batch_idx+1} 并行化重构完成")
            
            phase4_time = time.time() - phase4_start
            total_time = time.time() - start_time
            
            print(f"[Server {self.server_id}] 查询完成（多进程数据局部性优化）:")
            print(f"  阶段1 (多进程VDPF): {phase1_time:.2f}秒")
            print(f"  阶段2 (缓存友好e/f计算): {phase2_time:.2f}秒")
            print(f"  阶段3 (模拟交换): {phase3_time:.2f}秒") 
            print(f"  阶段4 (缓存友好重构): {phase4_time:.2f}秒")
            print(f"  总计: {total_time:.2f}秒")
            
            # 添加完成同步，确保所有服务器都读取完数据
            self._file_sync_barrier(query_id, "phase4_complete")
            
            # 现在可以安全清理文件了
            self._cleanup_query_files(query_id)
            
            # 返回结果
            print(f"[Server {self.server_id}] 准备构造响应...")
            
            # 安全的类型转换
            try:
                result_list = [int(x) % (2**32) for x in result_accumulator]
                print(f"[Server {self.server_id}] 结果转换成功，长度: {len(result_list)}")
                print(f"[Server {self.server_id}] DEBUG: 最终结果前5个值: {result_list[:5]}")
                print(f"[Server {self.server_id}] DEBUG: 最终结果范围: [{min(result_list)}, {max(result_list)}]")
            except Exception as e:
                print(f"[Server {self.server_id}] 结果转换失败: {e}")
                result_list = [0] * vector_dim  # 备用结果
            
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
                },
                'optimization_info': {
                    'cache_batch_size': self.cache_batch_size,
                    'total_batches': num_batches,
                    'vdpf_processes': self.vdpf_processes,
                    'avg_ops_per_sec_phase1': total_ops_per_sec
                }
            }
            
            print(f"[Server {self.server_id}] 响应数据构造完成")
            return response
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}
    
    def _file_sync_barrier(self, query_id: str, phase: str):
        """使用文件系统实现同步屏障"""
        # 创建本服务器的标记文件
        marker_file = f"server_{self.server_id}_query_{query_id}_{phase}_ready"
        marker_path = os.path.join(self.exchange_dir, marker_file)
        
        with open(marker_path, 'w') as f:
            f.write(str(time.time()))
        
        # 等待其他服务器的标记文件
        for other_id in [1, 2, 3]:
            if other_id == self.server_id:
                continue
            
            other_marker = f"server_{other_id}_query_{query_id}_{phase}_ready"
            other_path = os.path.join(self.exchange_dir, other_marker)
            
            max_wait = 60
            for i in range(max_wait):
                if os.path.exists(other_path):
                    break
                time.sleep(0.1)  # 减少轮询间隔
            
            if not os.path.exists(other_path):
                print(f"[Server {self.server_id}] 警告: Server {other_id} 未完成 {phase}")
    
    def _cleanup_query_files(self, query_id: str):
        """清理特定查询的文件"""
        for filename in os.listdir(self.exchange_dir):
            if query_id in filename and filename.endswith('_data.npz'):
                # 只清理数据文件，不清理同步标记文件
                try:
                    os.remove(os.path.join(self.exchange_dir, filename))
                except:
                    pass
    
    def _get_status(self) -> Dict:
        """获取服务器状态"""
        return {
            'status': 'success',
            'server_id': self.server_id,
            'mode': 'multiprocess_locality_optimized',
            'cache_batch_size': self.cache_batch_size,
            'vdpf_processes': self.vdpf_processes,
            'worker_threads': self.worker_threads,
            'data_loaded': {
                'nodes': self.node_shares.shape
            },
            'triples_available': self.mult_server.triple_array.shape[0] - self.mult_server.used_count if self.mult_server.triple_array is not None else 0,
            'triples_used': self.mult_server.used_count,
            'exchange_dir': self.exchange_dir
        }


def main():
    """主函数"""
    # 设置多进程启动方法为'spawn'，确保子进程正确初始化
    multiprocessing.set_start_method('spawn', force=True)
    
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='向量级多进程优化服务器')
    parser.add_argument('--server-id', type=int, required=True, choices=[1, 2, 3],
                        help='服务器ID (1, 2, 或 3)')
    parser.add_argument('--dataset', type=str, default='laion', choices=['laion', 'siftsmall', 'tripclick', 'ms_marco', 'nfcorpus'],
                        help='数据集名称 (默认: laion)')
    parser.add_argument('--vdpf-processes', type=int, default=4,
                        help='VDPF评估进程数 (默认: 4, 范围: 1-16)')
    
    args = parser.parse_args()
    
    if args.vdpf_processes < 1 or args.vdpf_processes > 16:
        print("错误: vdpf_processes 必须在 1-16 之间")
        sys.exit(1)
    
    print(f"启动服务器 {args.server_id}，数据集: {args.dataset}，使用 {args.vdpf_processes} 个VDPF评估进程")
    server = MemoryMappedOptimizedServer(args.server_id, args.dataset, args.vdpf_processes)
    server.start()


if __name__ == "__main__":
    main()