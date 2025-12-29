#!/usr/bin/env python3

import sys
import os
import socket
import json
import numpy as np
import time
import threading
from typing import Dict, List, Tuple
import hashlib
import concurrent.futures
from multiprocessing import Pool, cpu_count, Lock
import multiprocessing
import argparse
from multiprocessing import shared_memory

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('~/trident/standardDPF')
sys.path.append('~/trident/src')
sys.path.append('~/trident/query-opti')
sys.path.append('~/trident/query-opti')  # 添加优化目录

from dpf_wrapper import VDPFVectorWrapper
from binary_serializer import BinaryKeySerializer  # 导入二进制序列化器
from dpf_wrapper_optimized import OptimizedVDPFVectorWrapper  # 导入优化版wrapper
from binary_protocol import BinaryProtocol  # 导入二进制协议
from secure_multiplication import NumpyMultiplicationServer
from basic_functionalities import get_config, Share, MPC23SSS


# 全局函数，用于进程池调用（必须在模块顶层定义）
def warmup_process(process_id):
    """预热进程，加载必要的模块"""
    import time
    import sys
    sys.path.append('~/trident/query-opti')
    # 导入必要的模块
    from dpf_wrapper_optimized import OptimizedVDPFVectorWrapper
    from binary_serializer import BinaryKeySerializer
    return process_id

def evaluate_batch_range_process(args):
    """进程池工作函数：评估指定批次范围的VDPF"""
    process_id, start_batch, end_batch, cache_batch_size, num_nodes, serialized_key, server_id, shm_name, shape, dtype, dataset_name = args
    
    # CPU亲和性绑定
    try:
        import sys
        sys.path.append('~/trident/query-opti')
        from cpu_affinity_optimizer import set_process_affinity
        # 动态计算每个服务器应该使用的核心数
        total_cores = 16  # AMD EPYC 7R32的物理核心数
        num_servers = 3   # 总共3个服务器
        cores_per_server = total_cores // num_servers  # 每个服务器分配的核心数
        
        # 如果进程数超过了分配的核心数，使用循环分配
        if process_id < cores_per_server:
            set_process_affinity(server_id, process_id, cores_per_server)
        else:
            # 循环分配：process_id % cores_per_server
            effective_process_id = process_id % cores_per_server
            print(f"[Process {process_id}] 警告：进程数({process_id+1})超过核心数({cores_per_server})，循环绑定到核心")
            set_process_affinity(server_id, effective_process_id, cores_per_server)
    except Exception as e:
        print(f"[Process {process_id}] CPU绑定失败: {e}")
        import traceback
        traceback.print_exc()
    
    process_total_start = time.time()
    
    # 打印当前CPU亲和性
    import os
    current_cpus = os.sched_getaffinity(0)
    print(f"[Process {process_id}] 当前CPU亲和性: {sorted(current_cpus)}")
    
    # 计算实际要处理的节点数
    actual_nodes = 0
    for batch_idx in range(start_batch, end_batch):
        batch_start = batch_idx * cache_batch_size
        batch_end = min(batch_start + cache_batch_size, num_nodes)
        actual_nodes += (batch_end - batch_start)
    
    print(f"[Process {process_id}] 开始评估批次 {start_batch}-{end_batch-1} (实际节点数: {actual_nodes}) at {time.strftime('%H:%M:%S.%f')[:-3]}")
    
    # 时间测量1：VDPF实例创建
    t1 = time.time()
    from dpf_wrapper_optimized import OptimizedVDPFVectorWrapper
    dpf_wrapper = OptimizedVDPFVectorWrapper(dataset_name=dataset_name)
    vdpf_init_time = time.time() - t1
    
    # 时间测量2：密钥反序列化
    t2 = time.time()
    if isinstance(serialized_key, bytes):
        key = BinaryKeySerializer.deserialize_vdpf23_key(serialized_key)
    else:
        # 兼容旧的pickle格式
        key = dpf_wrapper._deserialize_key(serialized_key)
    deserialize_time = time.time() - t2
    
    # 时间测量3：连接共享内存
    t3 = time.time()
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    node_shares = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    shm_connect_time = time.time() - t3
    
    local_selector_shares = {}
    local_vector_shares = {}
    
    # 时间测量4：VDPF评估详细时间
    vdpf_eval_time = 0
    data_copy_time = 0
    batch_load_time = 0
    
    # VDPF内部时间细分
    vdpf_internal_time = {
        'eval_batch_calls': 0,
        'vdpf23_eval': 0,
        'vdpf_plus_eval': 0,
        'prg_operations': 0,
        'path_traversal': 0,
        'result_conversion': 0
    }
    
    for batch_idx in range(start_batch, end_batch):
        batch_start = batch_idx * cache_batch_size
        batch_end = min(batch_start + cache_batch_size, num_nodes)
        batch_size = batch_end - batch_start
        
        # 时间测量4a：批次数据加载
        t4a = time.time()
        batch_data = node_shares[batch_start:batch_end].copy()
        batch_load_time += time.time() - t4a
        
        # 时间测量4b：批量VDPF评估（带详细测量）
        t4b = time.time()
        
        # 临时修改eval_batch以收集内部时间
        batch_results = dpf_wrapper.eval_batch(key, batch_start, batch_end, server_id)
        
        vdpf_eval_time += time.time() - t4b
        
        # 时间测量4c：结果处理和数据复制
        t4c = time.time()
        for local_idx in range(batch_size):
            global_idx = batch_start + local_idx
            local_selector_shares[global_idx] = batch_results[global_idx]
            local_vector_shares[global_idx] = batch_data[local_idx]
        data_copy_time += time.time() - t4c
    
    process_total_time = time.time() - process_total_start
    # 使用之前计算的actual_nodes，更准确
    evaluated_nodes = actual_nodes
    
    # 详细时间分析
    print(f"\n[Process {process_id}] 详细时间分析 (完成于 {time.strftime('%H:%M:%S.%f')[:-3]}):")
    print(f"  - VDPF实例创建: {vdpf_init_time*1000:.1f}ms")
    print(f"  - 密钥反序列化: {deserialize_time*1000:.1f}ms")
    print(f"  - 共享内存连接: {shm_connect_time*1000:.1f}ms")
    print(f"  - 批次数据加载: {batch_load_time*1000:.1f}ms ({batch_load_time/process_total_time*100:.1f}%)")
    print(f"  - VDPF评估计算: {vdpf_eval_time*1000:.1f}ms ({vdpf_eval_time/process_total_time*100:.1f}%)")
    print(f"  - 数据复制操作: {data_copy_time*1000:.1f}ms ({data_copy_time/process_total_time*100:.1f}%)")
    print(f"  - 总耗时: {process_total_time*1000:.1f}ms")
    print(f"  - 节点数: {evaluated_nodes}, 速度: {evaluated_nodes/process_total_time:.0f} ops/sec")
    
    # 打印缓存统计（如果可用）
    try:
        if hasattr(dpf_wrapper.vdpf, 'get_cache_stats'):
            cache_stats = dpf_wrapper.vdpf.get_cache_stats()
            print(f"  - PRG缓存命中率: {cache_stats['hit_rate']:.1f}% (命中:{cache_stats['total_hits']}, 未命中:{cache_stats['total_misses']})")
    except:
        pass
    
    # 关闭共享内存连接
    existing_shm.close()
    
    return {
        'selector_shares': local_selector_shares,
        'vector_shares': local_vector_shares,
        'timing': {
            'vdpf_init': vdpf_init_time,
            'deserialize': deserialize_time,
            'shm_connect': shm_connect_time,
            'batch_load': batch_load_time,
            'vdpf_eval': vdpf_eval_time,
            'data_copy': data_copy_time,
            'total': process_total_time
        }
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
        self.dpf_wrapper = OptimizedVDPFVectorWrapper(dataset_name=dataset)
        self.mult_server = NumpyMultiplicationServer(server_id, self.config)
        
        # 加载数据
        self._load_data()
        
        # 模拟交换的目录（回退到标准文件系统）
        self.exchange_dir = "/tmp/mpc_exchange"
        os.makedirs(self.exchange_dir, exist_ok=True)
        
        # 清理旧的同步文件
        try:
            for filename in os.listdir(self.exchange_dir):
                if f"server_{self.server_id}_" in filename:
                    try:
                        os.remove(os.path.join(self.exchange_dir, filename))
                    except:
                        pass
            print(f"[Server {self.server_id}] 清理了旧的同步文件")
        except:
            pass
        
        # ===== 多进程优化参数 =====
        self.cache_batch_size = 1000  # 更小的批次以获得更好的负载均衡
        self.vdpf_processes = vdpf_processes  # 可配置的VDPF评估进程数
        self.worker_threads = 4  # 其他操作的线程数
        
        # 为其他操作创建线程池
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.worker_threads,
            thread_name_prefix=f"Server{server_id}-General"
        )
        
        # 创建持久的进程池，避免每次查询时的创建开销
        self.process_pool = Pool(processes=self.vdpf_processes)
        
        print(f"[Server {self.server_id}] 多进程数据局部性优化模式")
        print(f"[Server {self.server_id}] 缓存批次大小: {self.cache_batch_size}")
        print(f"[Server {self.server_id}] VDPF评估进程数: {self.vdpf_processes}")
        print(f"[Server {self.server_id}] 交换目录: {self.exchange_dir}")
        print(f"[Server {self.server_id}] 进程池已创建（{self.vdpf_processes}个进程）")
        
        # 预热进程池
        self._warmup_process_pool()
        
    def _warmup_process_pool(self):
        """预热进程池，确保所有进程都已启动并加载必要的模块"""
        print(f"[Server {self.server_id}] 预热进程池...")
        
        # 执行预热任务
        warmup_start = time.time()
        results = self.process_pool.map(warmup_process, range(self.vdpf_processes))
        warmup_time = time.time() - warmup_start
        
        print(f"[Server {self.server_id}] 进程池预热完成，耗时 {warmup_time:.2f}秒")
    
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
            # 关闭进程池
            if hasattr(self, 'process_pool'):
                print(f"[Server {self.server_id}] 关闭进程池...")
                self.process_pool.close()
                self.process_pool.join()
            
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
                # 先尝试读取4字节判断是否是二进制协议
                length_bytes = client_socket.recv(4)
                if not length_bytes:
                    break
                
                length = int.from_bytes(length_bytes, 'big')
                
                # 检查是否可能是二进制协议（通过长度判断）
                if length < 1000000:  # 合理的请求大小
                    # 读取第一个字节看是否是命令字节
                    first_byte = client_socket.recv(1)
                    if first_byte and first_byte[0] in [BinaryProtocol.CMD_QUERY_NODE_VECTOR, BinaryProtocol.CMD_GET_STATUS]:
                        # 是二进制协议
                        remaining = length - 1
                        data = first_byte
                        while len(data) < length:
                            chunk = client_socket.recv(min(remaining, 4096))
                            if not chunk:
                                break
                            data += chunk
                            remaining -= len(chunk)
                        
                        request = BinaryProtocol.decode_request(data)
                        print(f"[Server {self.server_id}] 收到二进制请求: {request.get('command', 'unknown')}")
                    else:
                        # JSON协议
                        data = first_byte
                        while len(data) < length:
                            chunk = client_socket.recv(min(length - len(data), 4096))
                            if not chunk:
                                break
                            data += chunk
                        
                        request = json.loads(data.decode())
                        print(f"[Server {self.server_id}] 收到JSON请求: {request.get('command', 'unknown')}")
                else:
                    # 长度异常，跳过
                    continue
                
                response = self._process_request(request)
                print(f"[Server {self.server_id}] 请求处理完成，响应状态: {response.get('status', 'unknown')}")
                
                # 使用二进制协议编码响应（保持原有结构）
                response_data = BinaryProtocol.encode_response(response)
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
        """多进程VDPF评估 - 使用共享内存优化"""
        
        # 创建共享内存以避免重复I/O
        shm_start = time.time()
        print(f"[Server {self.server_id}] 创建共享内存...")
        shm = shared_memory.SharedMemory(create=True, size=self.node_shares.nbytes)
        shared_array = np.ndarray(self.node_shares.shape, dtype=self.node_shares.dtype, buffer=shm.buf)
        shared_array[:] = self.node_shares[:]
        shm_time = time.time() - shm_start
        print(f"[Server {self.server_id}] 共享内存创建耗时: {shm_time:.3f}秒")
        
        # 改进的负载均衡分配算法
        # 计算每个进程应该处理的节点数，而不是批次数
        nodes_per_process = num_nodes // self.vdpf_processes
        remaining_nodes = num_nodes % self.vdpf_processes
        
        # 准备进程参数
        process_args = []
        current_node_start = 0
        
        for process_id in range(self.vdpf_processes):
            # 计算这个进程应该处理的节点数
            process_nodes = nodes_per_process + (1 if process_id < remaining_nodes else 0)
            
            if process_nodes == 0:
                continue
                
            # 计算这个进程的起始和结束节点
            node_start = current_node_start
            node_end = node_start + process_nodes
            
            # 转换为批次索引
            start_batch = node_start // self.cache_batch_size
            end_batch = (node_end + self.cache_batch_size - 1) // self.cache_batch_size
            
            args = (
                process_id,
                start_batch,
                min(end_batch, num_batches),
                self.cache_batch_size,
                num_nodes,
                serialized_key,
                self.server_id,
                shm.name,  # 传递共享内存名称
                self.node_shares.shape,  # 传递数组形状
                self.node_shares.dtype,  # 传递数据类型
                self.dataset  # 传递数据集名称
            )
            process_args.append(args)
            current_node_start = node_end
            
        # 打印负载分配信息
        print(f"[Server {self.server_id}] 负载均衡分配:")
        actual_node_counts = []
        for i, args in enumerate(process_args):
            process_id, start_batch, end_batch = args[0:3]
            batches = end_batch - start_batch
            # 计算实际节点数
            actual_nodes = 0
            for batch_idx in range(start_batch, end_batch):
                batch_start = batch_idx * self.cache_batch_size
                batch_end = min(batch_start + self.cache_batch_size, num_nodes)
                actual_nodes += (batch_end - batch_start)
            actual_node_counts.append(actual_nodes)
            print(f"  - Process {process_id}: 批次 {start_batch}-{end_batch-1} ({batches} 批次, {actual_nodes} 节点)")
        
        # 计算负载均衡统计
        if actual_node_counts:
            avg_nodes = np.mean(actual_node_counts)
            std_nodes = np.std(actual_node_counts)
            max_nodes = max(actual_node_counts)
            min_nodes = min(actual_node_counts)
            print(f"[Server {self.server_id}] 负载统计: 平均={avg_nodes:.0f}, 标准差={std_nodes:.1f}, 最大={max_nodes}, 最小={min_nodes}")
        
        # 使用持久的进程池并行执行
        print(f"[Server {self.server_id}] 使用持久进程池执行VDPF评估（{len(process_args)}个任务）")
        
        # 使用已创建的进程池
        results = self.process_pool.map(evaluate_batch_range_process, process_args)
        
        # 合并所有进程的结果
        all_selector_shares = {}
        all_vector_shares = {}
        process_timings = []
        
        for process_result in results:
            all_selector_shares.update(process_result['selector_shares'])
            all_vector_shares.update(process_result['vector_shares'])
            if 'timing' in process_result:
                process_timings.append(process_result['timing'])
        
        # 汇总时间分析
        if process_timings:
            print(f"\n[Server {self.server_id}] 阶段1时间分析汇总:")
            avg_vdpf_init = np.mean([t['vdpf_init'] for t in process_timings]) * 1000
            avg_deserialize = np.mean([t['deserialize'] for t in process_timings]) * 1000
            avg_shm_connect = np.mean([t['shm_connect'] for t in process_timings]) * 1000
            
            # 并行执行的实际时间是最慢进程的时间
            max_vdpf_eval = max([t['vdpf_eval'] for t in process_timings]) * 1000
            max_total_time = max([t['total'] for t in process_timings]) * 1000
            
            # CPU时间总和（用于了解总工作量）
            total_cpu_time = sum([t['total'] for t in process_timings]) * 1000
            total_vdpf_cpu = sum([t['vdpf_eval'] for t in process_timings]) * 1000
            
            print(f"  === 并行执行分析 ===")
            print(f"  - 最慢进程总时间: {max_total_time:.1f}ms (这是实际的并行执行时间)")
            print(f"  - 最慢进程VDPF时间: {max_vdpf_eval:.1f}ms")
            print(f"  - 并行效率: {total_cpu_time/max_total_time/self.vdpf_processes*100:.1f}%")
            print(f"  ")
            print(f"  === 开销分析 ===")
            print(f"  - 平均VDPF实例创建: {avg_vdpf_init:.1f}ms")
            print(f"  - 平均密钥反序列化: {avg_deserialize:.1f}ms")
            print(f"  - 平均共享内存连接: {avg_shm_connect:.1f}ms")
            print(f"  - 共享内存创建: {shm_time*1000:.1f}ms")
            print(f"  ")
            print(f"  === CPU时间统计 ===")
            print(f"  - 所有进程CPU时间总和: {total_cpu_time:.1f}ms")
            print(f"  - 所有进程VDPF计算总和: {total_vdpf_cpu:.1f}ms")
        
        print(f"[Server {self.server_id}] 多进程VDPF评估完成，共评估 {len(all_selector_shares)} 个节点")
        
        # 清理共享内存
        shm.close()
        shm.unlink()
        
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
        
        # print(f"[Server {self.server_id}] 已保存交换数据到 {filename}（无压缩优化）")
    
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
                # print(f"[Server {self.server_id}] 已加载 Server {other_id} 的数据（无压缩优化）")
                
                # 可选：验证数据完整性
                if 'hash' in data:
                    loaded_hash = str(data['hash'])
                    expected_hash = hashlib.md5((data['e_shares'].tobytes() + data['f_shares'].tobytes())).hexdigest()
                    if loaded_hash != expected_hash:
                        pass  # print(f"[Server {self.server_id}] 警告: Server {other_id} 数据校验和不匹配")
            else:
                pass  # print(f"[Server {self.server_id}] 警告: 无法找到 Server {other_id} 的数据")
        
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
            # print(f"[Server {self.server_id}] DEBUG: 向量维度: {vector_dim}, 查询ID: {query_id}")
            
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
            
            # # DEBUG: 检查VDPF选择器的值
            # non_zero_count = sum(1 for v in all_selector_shares.values() if v != 0)
            # print(f"[Server {self.server_id}] DEBUG: VDPF非零位置数: {non_zero_count}")
            # if non_zero_count > 0:
            #     for idx, val in list(all_selector_shares.items())[:5]:  # 打印前5个非零值
            #         if val != 0:
            #             print(f"[Server {self.server_id}] DEBUG: 位置 {idx} 的VDPF值: {val}")
            
            # 3.5. 文件同步屏障
            # print(f"[Server {self.server_id}] 使用文件系统同步...")
            self._file_sync_barrier(query_id, "phase1")
            
            # 4. 阶段2：数据局部性优化的e/f计算（保持原有实现）
            # print(f"[Server {self.server_id}] 阶段2: 缓存友好的e/f计算...")
            phase2_start = time.time()
            
            all_e_shares = np.zeros(num_nodes, dtype=np.uint64)
            all_f_shares = np.zeros((num_nodes, vector_dim), dtype=np.uint64)
            all_computation_states = {}
            
            # 同样使用批次处理，避免随机内存访问
            for batch_idx in range(num_batches):
                batch_start = batch_idx * self.cache_batch_size
                batch_end = min(batch_start + self.cache_batch_size, num_nodes)
                batch_size = batch_end - batch_start
                
                # print(f"[Server {self.server_id}] e/f计算批次 {batch_idx+1}/{num_batches}")
                
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
            # print(f"[Server {self.server_id}] 阶段2完成，耗时 {phase2_time:.2f}秒")
            
            # 5. 阶段3：模拟批量交换（保持无压缩优化）
            # print(f"[Server {self.server_id}] 阶段3: 模拟批量交换...")
            phase3_start = time.time()
            
            # 保存本服务器的数据
            self._save_exchange_data(query_id, all_e_shares, all_f_shares)
            
            # 等待其他服务器保存完成
            self._file_sync_barrier(query_id, "phase3_save")
            
            # 读取其他服务器的数据
            all_e_from_others, all_f_from_others = self._load_other_servers_data(query_id, num_nodes)
            
            phase3_time = time.time() - phase3_start
            # print(f"[Server {self.server_id}] 阶段3完成（模拟），耗时 {phase3_time:.2f}秒")
            
            # 6. 阶段4：数据局部性优化的重构计算（保持NumPy并行化优化）
            # print(f"[Server {self.server_id}] 阶段4: 缓存友好的重构计算...")
            phase4_start = time.time()
            
            # 预计算拉格朗日系数，用于所有批次
            lagrange_1 = 2
            lagrange_2 = self.field_size - 1
            
            # 使用批次处理进行并行化重构计算
            for batch_idx in range(num_batches):
                batch_start = batch_idx * self.cache_batch_size
                batch_end = min(batch_start + self.cache_batch_size, num_nodes)
                batch_size = batch_end - batch_start
                
                # print(f"[Server {self.server_id}] NumPy并行化重构批次 {batch_idx+1}/{num_batches} ({batch_size} 节点)")
                
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
                
                # # DEBUG: 第一个批次的调试信息
                # if batch_idx == 0:
                #     print(f"[Server {self.server_id}] DEBUG: 批次1贡献的前5个值: {batch_contribution[:5]}")
                #     print(f"[Server {self.server_id}] DEBUG: 累加器的前5个值: {result_accumulator[:5]}")
                
                # 9. 清理计算缓存
                for local_idx in range(batch_size):
                    global_idx = batch_start + local_idx
                    state = all_computation_states[global_idx]
                    computation_id = state['computation_id']
                    if computation_id in self.mult_server.computation_cache:
                        del self.mult_server.computation_cache[computation_id]
                
                # print(f"[Server {self.server_id}] 批次 {batch_idx+1} 并行化重构完成")
            
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
            
            # 延迟清理，给其他服务器时间完成同步
            # 只清理数据文件，同步文件会在下次查询开始时清理
            self._cleanup_data_files_only(query_id)
            
            # 返回结果
            # print(f"[Server {self.server_id}] 准备构造响应...")
            
            # 安全的类型转换
            try:
                result_list = [int(x) % (2**32) for x in result_accumulator]
                # print(f"[Server {self.server_id}] 结果转换成功，长度: {len(result_list)}")
                # print(f"[Server {self.server_id}] DEBUG: 最终结果前5个值: {result_list[:5]}")
                # print(f"[Server {self.server_id}] DEBUG: 最终结果范围: [{min(result_list)}, {max(result_list)}]")
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
            
            # print(f"[Server {self.server_id}] 响应数据构造完成")
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
            if query_id in filename:
                # 清理数据文件和同步标记文件
                try:
                    os.remove(os.path.join(self.exchange_dir, filename))
                except:
                    pass
    
    def _cleanup_data_files_only(self, query_id: str):
        """只清理数据文件，保留同步文件"""
        for filename in os.listdir(self.exchange_dir):
            if query_id in filename and filename.endswith('_data.npz'):
                # 只清理数据文件
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