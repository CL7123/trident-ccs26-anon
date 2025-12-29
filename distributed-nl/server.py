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
import logging

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('~/trident/standardDPF')
sys.path.append('~/trident/src')
sys.path.append('~/trident/query-opti')

from dpf_wrapper import VDPFVectorWrapper
from binary_serializer import BinaryKeySerializer
from dpf_wrapper_optimized import OptimizedVDPFVectorWrapper
from binary_protocol import BinaryProtocol
from secure_multiplication import NumpyMultiplicationServer
from basic_functionalities import get_config, Share, MPC23SSS
from config import SERVERS, SERVER_TO_SERVER

# 设置日志
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [NL-Server] %(message)s')
logger = logging.getLogger(__name__)

# 全局函数，用于进程池调用
def warmup_process(process_id):
    """预热进程，加载必要的模块"""
    import time
    import sys
    sys.path.append('~/trident/query-opti')
    from dpf_wrapper_optimized import OptimizedVDPFVectorWrapper
    from binary_serializer import BinaryKeySerializer
    return process_id

def evaluate_batch_range_process(args):
    """进程池工作函数：评估指定批次范围的VDPF用于邻居列表"""
    process_id, start_batch, end_batch, cache_batch_size, num_nodes, serialized_key, server_id, shm_name, shape, dtype, dataset_name = args
    
    # CPU亲和性绑定
    try:
        import sys
        sys.path.append('~/trident/query-opti')
        from cpu_affinity_optimizer import set_process_affinity
        total_cores = cpu_count()
        
        if process_id < total_cores:
            import os
            pid = os.getpid()
            os.sched_setaffinity(pid, {process_id})
            print(f"[NL-Server {server_id}, Process {process_id}] 绑定到核心 {process_id}")
        else:
            core_id = process_id % total_cores
            import os
            pid = os.getpid()
            os.sched_setaffinity(pid, {core_id})
            print(f"[NL-Server {server_id}, Process {process_id}] 绑定到核心 {core_id}")
    except Exception as e:
        print(f"[Process {process_id}] CPU绑定失败: {e}")
    
    process_total_start = time.time()
    
    # 计算实际要处理的邻居列表数
    actual_lists = 0
    for batch_idx in range(start_batch, end_batch):
        batch_start = batch_idx * cache_batch_size
        batch_end = min(batch_start + cache_batch_size, num_nodes)
        actual_lists += (batch_end - batch_start)
    
    # VDPF实例创建
    from dpf_wrapper_optimized import OptimizedVDPFVectorWrapper
    dpf_wrapper = OptimizedVDPFVectorWrapper(dataset_name=dataset_name)
    
    # 密钥反序列化
    if isinstance(serialized_key, bytes):
        key = BinaryKeySerializer.deserialize_vdpf23_key(serialized_key)
    else:
        key = dpf_wrapper._deserialize_key(serialized_key)
    
    # 连接共享内存（邻居列表数据）
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    neighbor_shares = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    
    local_selector_shares = {}
    local_neighbor_shares = {}
    
    # VDPF评估 - 用于邻居列表查询
    for batch_idx in range(start_batch, end_batch):
        batch_start = batch_idx * cache_batch_size
        batch_end = min(batch_start + cache_batch_size, num_nodes)
        batch_size = batch_end - batch_start
        
        batch_data = neighbor_shares[batch_start:batch_end].copy()
        batch_results = dpf_wrapper.eval_batch(key, batch_start, batch_end, server_id)
        
        for local_idx in range(batch_size):
            global_idx = batch_start + local_idx
            local_selector_shares[global_idx] = batch_results[global_idx]
            local_neighbor_shares[global_idx] = batch_data[local_idx]
    
    # 关闭共享内存连接
    existing_shm.close()
    
    return {
        'selector_shares': local_selector_shares,
        'neighbor_shares': local_neighbor_shares
    }


class DistributedNeighborServer:
    """分布式邻居列表查询服务器"""
    
    def __init__(self, server_id: int, dataset: str = "siftsmall", vdpf_processes: int = 32):
        self.server_id = server_id
        self.dataset = dataset
        self.config = get_config(dataset)
        self.field_size = self.config.prime
        self.mpc = MPC23SSS(self.config)
        
        # 网络配置 - 监听所有接口
        self.host = "0.0.0.0"  # 接受所有外部连接
        self.port = 9000 + server_id  # 使用9001-9003端口，避免与向量查询服务冲突
        
        # 初始化组件
        self.dpf_wrapper = OptimizedVDPFVectorWrapper(dataset_name=dataset)
        self.mult_server = NumpyMultiplicationServer(server_id, self.config)
        
        # 加载邻居列表数据
        self._load_neighbor_data()
        
        # 交换目录
        self.exchange_dir = "/tmp/mpc_nl_exchange"
        os.makedirs(self.exchange_dir, exist_ok=True)
        
        # 清理旧文件
        self._cleanup_old_files()
        
        # 多进程优化参数
        self.cache_batch_size = max(100, 1000 // max(vdpf_processes // 4, 1))
        self.vdpf_processes = vdpf_processes
        self.worker_threads = 8
        logger.info(f"使用批处理大小: {self.cache_batch_size}, VDPF进程数: {self.vdpf_processes}")
        
        # 创建线程池
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.worker_threads,
            thread_name_prefix=f"NLServer{server_id}-General"
        )
        
        # 创建进程池
        self.process_pool = Pool(processes=self.vdpf_processes)
        
        # 数据交换存储
        self.exchange_storage = {}  # {query_id: {'e_shares': ..., 'f_shares': ...}}
        
        # 服务器间连接管理
        self.server_connections = {}  # {server_id: socket}
        self.server_config = SERVER_TO_SERVER  # 服务器间通信使用私网IP
        
        # 网络传输统计
        self.network_stats = {
            'total_bytes_sent': 0,
            'total_bytes_received': 0,
            'transfer_count': 0,
            'transfer_details': []
        }
        
        logger.info(f"分布式邻居列表服务器初始化完成")
        logger.info(f"监听地址: {self.host}:{self.port}")
        logger.info(f"数据集: {self.dataset}")
        logger.info(f"VDPF进程数: {self.vdpf_processes}")
        
        # 预热进程池
        self._warmup_process_pool()
        
        # 延迟建立连接
        self.connections_established = False
        
    def _cleanup_old_files(self):
        """清理旧的交换文件"""
        try:
            for filename in os.listdir(self.exchange_dir):
                if f"nl_server_{self.server_id}_" in filename:
                    try:
                        os.remove(os.path.join(self.exchange_dir, filename))
                    except:
                        pass
            logger.info("清理了旧的同步文件")
        except:
            pass
    
    def _warmup_process_pool(self):
        """预热进程池"""
        logger.info("预热进程池...")
        warmup_start = time.time()
        results = self.process_pool.map(warmup_process, range(self.vdpf_processes))
        warmup_time = time.time() - warmup_start
        logger.info(f"进程池预热完成，耗时 {warmup_time:.2f}秒")
    
    def _load_neighbor_data(self):
        """加载邻居列表秘密共享数据"""
        logger.info(f"加载{self.dataset}邻居列表数据...")
        
        self.data_dir = f"~/trident/dataset/{self.dataset}/server_{self.server_id}"
        
        # 加载邻居列表份额
        self.neighbors_path = os.path.join(self.data_dir, "neighbors_shares.npy")
        self.neighbor_shares = np.load(self.neighbors_path)
        logger.info(f"邻居列表数据: {self.neighbor_shares.shape}")
        logger.info(f"数据大小: {self.neighbor_shares.nbytes / 1024 / 1024:.1f}MB")
        logger.info(f"数据类型: {self.neighbor_shares.dtype}")
        
        # 邻居列表的第二维是K（最近邻数量）
        self.num_nodes = self.neighbor_shares.shape[0]
        self.k_neighbors = self.neighbor_shares.shape[1] if len(self.neighbor_shares.shape) > 1 else 1
        logger.info(f"节点数: {self.num_nodes}, 每个节点的邻居数: {self.k_neighbors}")
        
        logger.info(f"三元组可用: {self.mult_server.triple_array.shape[0] - self.mult_server.used_count if self.mult_server.triple_array is not None else 0}")
    
    def start(self):
        """启动服务器"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # 设置socket选项以改善网络性能
        server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        
        try:
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)
            logger.info(f"邻居列表服务器启动成功，监听 {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"无法绑定到 {self.host}:{self.port}: {e}")
            return
        
        try:
            while True:
                client_socket, address = server_socket.accept()
                logger.info(f"接受连接来自 {address}")
                
                # 设置客户端socket选项
                client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address)
                )
                client_thread.daemon = True
                client_thread.start()
                
        except KeyboardInterrupt:
            logger.info("接收到中断信号，正在关闭服务器...")
        except Exception as e:
            logger.error(f"服务器错误: {e}")
        finally:
            server_socket.close()
            self._cleanup_resources()
            logger.info("服务器已关闭")
    
    def _cleanup_resources(self):
        """清理资源"""
        self._cleanup_old_files()
        self.executor.shutdown(wait=True)
        if hasattr(self, 'process_pool'):
            logger.info("关闭进程池...")
            self.process_pool.close()
            self.process_pool.join()
    
    def _handle_client(self, client_socket: socket.socket, address):
        """处理客户端请求"""
        try:
            while True:
                # 读取请求长度
                length_bytes = client_socket.recv(4)
                if not length_bytes:
                    break
                
                length = int.from_bytes(length_bytes, 'big')
                
                # 读取请求数据
                data = b''
                while len(data) < length:
                    chunk = client_socket.recv(min(length - len(data), 4096))
                    if not chunk:
                        break
                    data += chunk
                
                if len(data) < length:
                    logger.warning(f"从 {address} 接收到不完整的数据")
                    break
                
                # 解析请求
                try:
                    # 检查是否是二进制协议
                    first_byte = data[0] if data else -1
                    logger.info(f"接收到的第一个字节: {first_byte}, 数据长度: {len(data)}")
                    logger.info(f"期望的命令字节: NODE_VECTOR={BinaryProtocol.CMD_QUERY_NODE_VECTOR}, STATUS={BinaryProtocol.CMD_GET_STATUS}, NEIGHBOR_LIST={BinaryProtocol.CMD_QUERY_NEIGHBOR_LIST}")
                    
                    if first_byte in [BinaryProtocol.CMD_QUERY_NODE_VECTOR, BinaryProtocol.CMD_GET_STATUS, BinaryProtocol.CMD_QUERY_NEIGHBOR_LIST]:
                        request = BinaryProtocol.decode_request(data)
                        logger.info(f"收到二进制请求: {request.get('command', 'unknown')}")
                    else:
                        # 尝试作为JSON解析
                        logger.info(f"尝试JSON解析，前20字节: {data[:20]}")
                        request = json.loads(data.decode())
                        logger.info(f"收到JSON请求: {request.get('command', 'unknown')}")
                except Exception as e:
                    logger.error(f"解析请求失败: {e}")
                    continue
                
                # 特殊处理二进制数据交换
                if request.get('command') == 'binary_exchange_data':
                    receive_start_time = time.time()
                    from_server = request.get('from_server')
                    query_id = request.get('query_id')
                    
                    logger.info(f"接收来自服务器 {from_server} 的二进制数据交换 (查询: {query_id})")
                    logger.info(f"  期望形状: e_shares={request.get('e_shares_shape')}, f_shares={request.get('f_shares_shape')}")
                    
                    # 接收e_shares - 使用MSG_WAITALL消除缓冲区循环
                    e_len_bytes = client_socket.recv(4, socket.MSG_WAITALL)
                    if len(e_len_bytes) != 4:
                        logger.error(f"接收e_shares长度头失败: {len(e_len_bytes)} bytes")
                        continue
                    
                    e_len = int.from_bytes(e_len_bytes, 'big')
                    logger.info(f"  接收 e_shares: {e_len:,} bytes")
                    
                    e_data = client_socket.recv(e_len, socket.MSG_WAITALL)
                    if len(e_data) != e_len:
                        logger.error(f"e_shares接收不完整: {len(e_data)}/{e_len} bytes")
                        continue
                    
                    # 接收f_shares - 使用MSG_WAITALL消除缓冲区循环
                    f_len_bytes = client_socket.recv(4, socket.MSG_WAITALL)
                    if len(f_len_bytes) != 4:
                        logger.error(f"接收f_shares长度头失败: {len(f_len_bytes)} bytes")
                        continue
                    
                    f_len = int.from_bytes(f_len_bytes, 'big')
                    logger.info(f"  接收 f_shares: {f_len:,} bytes")
                    
                    f_data = client_socket.recv(f_len, socket.MSG_WAITALL)
                    if len(f_data) != f_len:
                        logger.error(f"f_shares接收不完整: {len(f_data)}/{f_len} bytes")
                        continue
                    
                    total_received_size = len(data) + len(e_data) + len(f_data) + 8  # +8 for length headers
                    logger.info(f"  总接收: {total_received_size:,} bytes ({total_received_size/1024/1024:.2f} MB)")
                    
                    # 重构数组
                    e_shares = np.frombuffer(e_data, dtype=np.uint64)
                    f_shape = tuple(request['f_shares_shape'])
                    f_shares = np.frombuffer(f_data, dtype=np.uint64).reshape(f_shape)
                    
                    # 存储数据
                    if query_id not in self.exchange_storage:
                        self.exchange_storage[query_id] = {}
                    
                    self.exchange_storage[query_id][f'e_shares_{from_server}'] = e_shares
                    self.exchange_storage[query_id][f'f_shares_{from_server}'] = f_shares
                    
                    # 计算接收统计
                    receive_duration = time.time() - receive_start_time
                    receive_speed = total_received_size / receive_duration if receive_duration > 0 else 0
                    
                    # 更新统计信息
                    self.network_stats['total_bytes_received'] += total_received_size
                    
                    logger.info(f"数据接收完成:")
                    logger.info(f"  来源: 服务器 {from_server}")
                    logger.info(f"  耗时: {receive_duration:.3f}秒")
                    logger.info(f"  速度: {receive_speed/1024/1024:.2f} MB/s ({(receive_speed*8)/1024/1024:.2f} Mbps)")
                    
                    # 发送确认响应
                    response = {'status': 'success'}
                    response_data = json.dumps(response).encode()
                    client_socket.sendall(len(response_data).to_bytes(4, 'big'))
                    client_socket.sendall(response_data)
                    continue
                
                # 处理其他请求
                response = self._process_request(request)
                
                # 发送响应
                response_data = BinaryProtocol.encode_response(response)
                client_socket.sendall(response_data)
                
        except ConnectionResetError:
            logger.info(f"客户端 {address} 断开连接")
        except Exception as e:
            logger.error(f"处理客户端 {address} 时出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            client_socket.close()
    
    def _process_request(self, request: Dict) -> Dict:
        """处理请求"""
        command = request.get('command')
        
        if command == 'query_neighbor_list':
            return self._handle_neighbor_list_query(request)
        elif command == 'get_status':
            return self._get_status()
        elif command == 'exchange_data':
            return self._handle_data_exchange(request)
        elif command == 'binary_exchange_data':
            return self._handle_binary_exchange_data(request)
        elif command == 'establish_connections':
            self._establish_persistent_connections()
            return {'status': 'success', 'connections': len(self.server_connections)}
        else:
            return {'status': 'error', 'message': f'未知命令: {command}'}
    
    def _establish_persistent_connections(self):
        """建立到其他服务器的持久连接"""
        logger.info("建立持久连接到其他服务器...")
        
        for server_id, server_info in self.server_config.items():
            if server_id == self.server_id or server_id in self.server_connections:
                continue
            
            # 尝试建立连接到邻居列表服务器端口（9001-9003）
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024 * 1024)
                sock.settimeout(1)
                # 使用邻居列表服务器端口
                sock.connect((server_info['host'], 9000 + server_id))
                self.server_connections[server_id] = sock
                logger.info(f"成功建立持久连接到邻居列表服务器 {server_id}")
            except Exception as e:
                logger.debug(f"邻居列表服务器 {server_id} 尚未就绪: {e}")
    
    def _send_binary_exchange_data(self, target_server_id: int, query_id: str, e_shares: np.ndarray, f_shares: np.ndarray) -> bool:
        """发送二进制格式的交换数据"""
        send_start_time = time.time()
        
        # 第一次调用时建立所有连接
        if not self.connections_established:
            logger.info("第一次数据交换，建立持久连接...")
            self._establish_persistent_connections()
            self.connections_established = True
        
        # 检查连接是否存在且有效
        if target_server_id not in self.server_connections:
            logger.warning(f"没有到服务器 {target_server_id} 的持久连接，尝试建立新连接...")
            # 尝试建立连接
            try:
                server_info = self.server_config[target_server_id]
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024 * 1024)
                sock.settimeout(30)
                # 连接到邻居列表服务器端口
                sock.connect((server_info['host'], 9000 + target_server_id))
                self.server_connections[target_server_id] = sock
                logger.info(f"成功建立新连接到邻居列表服务器 {target_server_id}")
            except Exception as e:
                logger.error(f"无法连接到邻居列表服务器 {target_server_id}: {e}")
                return False
        else:
            logger.debug(f"使用现有持久连接到服务器 {target_server_id}")
        
        try:
            sock = self.server_connections[target_server_id]
            
            # 准备元数据
            metadata = {
                'command': 'binary_exchange_data',
                'query_id': query_id,
                'from_server': self.server_id,
                'e_shares_shape': e_shares.shape,
                'f_shares_shape': f_shares.shape
            }
            
            # 计算数据大小
            e_bytes = e_shares.tobytes()
            f_bytes = f_shares.tobytes()
            metadata_bytes = json.dumps(metadata).encode()
            
            # 发送元数据
            sock.sendall(len(metadata_bytes).to_bytes(4, 'big'))
            sock.sendall(metadata_bytes)
            
            total_data_size = len(metadata_bytes) + len(e_bytes) + len(f_bytes) + 12  # +12 for length headers
            
            logger.info(f"发送数据到邻居列表服务器 {target_server_id}:")
            logger.info(f"  e_shares: {e_shares.shape}, {len(e_bytes):,} bytes")
            logger.info(f"  f_shares: {f_shares.shape}, {len(f_bytes):,} bytes")
            logger.info(f"  元数据: {len(metadata_bytes):,} bytes")
            logger.info(f"  总大小: {total_data_size:,} bytes ({total_data_size/1024/1024:.2f} MB)")
            
            # 发送e_shares
            sock.sendall(len(e_bytes).to_bytes(4, 'big'))
            sock.sendall(e_bytes)
            
            # 发送f_shares
            sock.sendall(len(f_bytes).to_bytes(4, 'big'))
            sock.sendall(f_bytes)
            
            # 接收确认
            response_len = int.from_bytes(sock.recv(4), 'big')
            response_data = sock.recv(response_len)
            response = json.loads(response_data.decode())
            
            # 计算传输统计
            send_duration = time.time() - send_start_time
            transfer_speed = total_data_size / send_duration if send_duration > 0 else 0
            
            # 更新统计信息
            self.network_stats['total_bytes_sent'] += total_data_size
            self.network_stats['transfer_count'] += 1
            self.network_stats['transfer_details'].append({
                'target_server': target_server_id,
                'query_id': query_id,
                'data_size_bytes': total_data_size,
                'data_size_mb': total_data_size / 1024 / 1024,
                'duration_seconds': send_duration,
                'speed_mbps': (transfer_speed * 8) / (1024 * 1024),
                'speed_mb_per_sec': transfer_speed / (1024 * 1024),
                'timestamp': time.time()
            })
            
            logger.info(f"数据传输完成:")
            logger.info(f"  耗时: {send_duration:.3f}秒")
            logger.info(f"  速度: {transfer_speed/1024/1024:.2f} MB/s ({(transfer_speed*8)/1024/1024:.2f} Mbps)")
            
            return response.get('status') == 'success'
            
        except Exception as e:
            logger.error(f"发送二进制数据到服务器 {target_server_id} 失败: {e}")
            # 关闭失效的连接
            if target_server_id in self.server_connections:
                try:
                    self.server_connections[target_server_id].close()
                except:
                    pass
                del self.server_connections[target_server_id]
            return False
    
    def _exchange_data_with_servers(self, query_id: str, e_shares: np.ndarray, f_shares: np.ndarray) -> tuple:
        """与其他服务器交换数据"""
        send_start = time.time()
        
        # 并行向其他服务器发送数据
        def send_to_server_async(server_id):
            if server_id == self.server_id:
                return None
            start = time.time()
            success = self._send_binary_exchange_data(server_id, query_id, e_shares, f_shares)
            elapsed = time.time() - start
            if success:
                logger.info(f"成功向服务器 {server_id} 发送二进制数据 (耗时: {elapsed:.3f}秒)")
            else:
                logger.error(f"向服务器 {server_id} 发送二进制数据失败")
            return server_id, success
        
        # 使用线程池并行发送
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(send_to_server_async, sid) for sid in [1, 2, 3] if sid != self.server_id]
            concurrent.futures.wait(futures)
        
        send_time = time.time() - send_start
        logger.info(f"发送数据耗时: {send_time:.3f}秒")
        
        # 等待接收其他服务器的数据
        max_wait = 30  # 最多等待30秒
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            # 检查是否收到所有其他服务器的数据
            received_all = True
            for server_id in [1, 2, 3]:
                if server_id == self.server_id:
                    continue
                
                if query_id not in self.exchange_storage or \
                   f'e_shares_{server_id}' not in self.exchange_storage[query_id]:
                    received_all = False
                    break
            
            if received_all:
                # 整理数据
                all_e_from_others = {}
                all_f_from_others = {}
                
                for server_id in [1, 2, 3]:
                    if server_id == self.server_id:
                        continue
                    
                    all_e_from_others[server_id] = self.exchange_storage[query_id][f'e_shares_{server_id}']
                    all_f_from_others[server_id] = self.exchange_storage[query_id][f'f_shares_{server_id}']
                
                return all_e_from_others, all_f_from_others
            
            time.sleep(0.1)
        
        logger.error(f"等待其他服务器数据超时 (查询: {query_id})")
        return {}, {}
    
    def _multiprocess_vdpf_evaluation(self, serialized_key, num_nodes, num_batches):
        """多进程VDPF评估 - 用于邻居列表"""
        # 创建共享内存
        shm = shared_memory.SharedMemory(create=True, size=self.neighbor_shares.nbytes)
        shared_array = np.ndarray(self.neighbor_shares.shape, dtype=self.neighbor_shares.dtype, buffer=shm.buf)
        shared_array[:] = self.neighbor_shares[:]
        
        # 负载均衡分配
        nodes_per_process = num_nodes // self.vdpf_processes
        remaining_nodes = num_nodes % self.vdpf_processes
        
        # 准备进程参数
        process_args = []
        current_node_start = 0
        
        for process_id in range(self.vdpf_processes):
            process_nodes = nodes_per_process + (1 if process_id < remaining_nodes else 0)
            
            if process_nodes == 0:
                continue
                
            node_start = current_node_start
            node_end = node_start + process_nodes
            
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
                shm.name,
                self.neighbor_shares.shape,
                self.neighbor_shares.dtype,
                self.dataset
            )
            process_args.append(args)
            current_node_start = node_end
        
        # 使用进程池执行
        results = self.process_pool.map(evaluate_batch_range_process, process_args)
        
        # 合并结果
        all_selector_shares = {}
        all_neighbor_shares = {}
        
        for process_result in results:
            all_selector_shares.update(process_result['selector_shares'])
            all_neighbor_shares.update(process_result['neighbor_shares'])
        
        # 清理共享内存
        shm.close()
        shm.unlink()
        
        return all_selector_shares, all_neighbor_shares
    
    def _handle_neighbor_list_query(self, request: Dict) -> Dict:
        """处理邻居列表查询"""
        try:
            serialized_key = request['dpf_key']
            query_id = request.get('query_id', 'unknown')
            
            logger.info(f"处理邻居列表查询，查询ID: {query_id}")
            
            # 反序列化密钥
            key = self.dpf_wrapper._deserialize_key(serialized_key)
            
            # 初始化
            start_time = time.time()
            num_nodes = len(self.neighbor_shares)
            result_accumulator = np.zeros(self.k_neighbors, dtype=np.int64)
            
            # 阶段1：多进程VDPF评估
            logger.info(f"阶段1: 多进程VDPF评估 ({self.vdpf_processes} 进程)...")
            phase1_start = time.time()
            
            num_batches = (num_nodes + self.cache_batch_size - 1) // self.cache_batch_size
            
            all_selector_shares, all_neighbor_shares = self._multiprocess_vdpf_evaluation(
                serialized_key, num_nodes, num_batches)
            
            phase1_time = time.time() - phase1_start
            logger.info(f"阶段1完成，耗时 {phase1_time:.2f}秒")
            
            # 阶段2：e/f计算
            logger.info("阶段2: e/f计算...")
            phase2_start = time.time()
            
            all_e_shares = np.zeros(num_nodes, dtype=np.uint64)
            all_f_shares = np.zeros((num_nodes, self.k_neighbors), dtype=np.uint64)
            all_computation_states = {}
            
            logger.info(f"初始化 e_shares: dtype={all_e_shares.dtype}, shape={all_e_shares.shape}")
            logger.info(f"初始化 f_shares: dtype={all_f_shares.dtype}, shape={all_f_shares.shape}")
            
            for batch_idx in range(num_batches):
                batch_start = batch_idx * self.cache_batch_size
                batch_end = min(batch_start + self.cache_batch_size, num_nodes)
                batch_size = batch_end - batch_start
                
                # 批量获取三元组
                batch_triples = []
                for _ in range(batch_size):
                    batch_triples.append(self.mult_server.get_next_triple())
                
                # 处理批次
                for local_idx in range(batch_size):
                    global_idx = batch_start + local_idx
                    computation_id = f"nl_query_{query_id}_pos{global_idx}"
                    
                    a, b, c = batch_triples[local_idx]
                    
                    e_share = (all_selector_shares[global_idx] - a) % self.field_size
                    all_e_shares[global_idx] = e_share
                    
                    neighbor_data = all_neighbor_shares[global_idx]
                    f_values = (neighbor_data.astype(np.int64) - b) % self.field_size
                    all_f_shares[global_idx] = f_values
                    
                    all_computation_states[global_idx] = {
                        'selector_share': all_selector_shares[global_idx],
                        'neighbor_share': all_neighbor_shares[global_idx],
                        'computation_id': computation_id,
                        'triple': (a, b, c)
                    }
                    
                    self.mult_server.computation_cache[computation_id] = {
                        'a': a,
                        'b': b,
                        'c': c
                    }
            
            phase2_time = time.time() - phase2_start
            logger.info(f"阶段2完成，耗时 {phase2_time:.2f}秒")
            
            # 阶段3：数据交换
            logger.info("阶段3: 数据交换...")
            phase3_start = time.time()
            
            # 通过网络与其他服务器交换数据
            all_e_from_others, all_f_from_others = self._exchange_data_with_servers(query_id, all_e_shares, all_f_shares)
            
            phase3_time = time.time() - phase3_start
            logger.info(f"阶段3完成，耗时 {phase3_time:.2f}秒")
            
            # 阶段4：重构计算
            logger.info("阶段4: 重构计算...")
            phase4_start = time.time()
            
            # 拉格朗日系数
            lagrange_1 = 2
            lagrange_2 = self.field_size - 1
            
            # 批量重构
            for batch_idx in range(num_batches):
                batch_start = batch_idx * self.cache_batch_size
                batch_end = min(batch_start + self.cache_batch_size, num_nodes)
                batch_size = batch_end - batch_start
                
                # 提取批次数据
                batch_e_shares_local = all_e_shares[batch_start:batch_end]
                batch_f_shares_local = all_f_shares[batch_start:batch_end, :]
                
                # 构建份额矩阵
                e_shares_matrix = np.zeros((batch_size, 3), dtype=np.uint64)
                e_shares_matrix[:, self.server_id - 1] = batch_e_shares_local
                
                f_shares_matrix = np.zeros((batch_size, self.k_neighbors, 3), dtype=np.uint64)
                f_shares_matrix[:, :, self.server_id - 1] = batch_f_shares_local
                
                # 填入其他服务器数据
                for other_id, other_e_shares in all_e_from_others.items():
                    e_shares_matrix[:, other_id - 1] = other_e_shares[batch_start:batch_end]
                
                for other_id, other_f_shares in all_f_from_others.items():
                    f_shares_matrix[:, :, other_id - 1] = other_f_shares[batch_start:batch_end, :]
                
                # 重构
                batch_e_reconstructed = (e_shares_matrix[:, 0] * lagrange_1 + 
                                       e_shares_matrix[:, 1] * lagrange_2) % self.field_size
                
                batch_f_reconstructed = (f_shares_matrix[:, :, 0] * lagrange_1 + 
                                       f_shares_matrix[:, :, 1] * lagrange_2) % self.field_size
                
                # 获取三元组
                batch_triples = []
                for local_idx in range(batch_size):
                    global_idx = batch_start + local_idx
                    state = all_computation_states[global_idx]
                    a, b, c = state['triple']
                    batch_triples.append((a, b, c))
                
                batch_a = np.array([t[0] for t in batch_triples], dtype=np.uint64)
                batch_b = np.array([t[1] for t in batch_triples], dtype=np.uint64)
                batch_c = np.array([t[2] for t in batch_triples], dtype=np.uint64)
                
                # 计算结果 - 对邻居列表的每个元素进行计算
                batch_e_expanded = batch_e_reconstructed[:, np.newaxis]
                batch_a_expanded = batch_a[:, np.newaxis]
                batch_b_expanded = batch_b[:, np.newaxis]
                batch_c_expanded = batch_c[:, np.newaxis]
                
                batch_result = batch_c_expanded
                batch_result = (batch_result + batch_e_expanded * batch_b_expanded) % self.field_size
                batch_result = (batch_result + batch_f_reconstructed * batch_a_expanded) % self.field_size
                batch_result = (batch_result + batch_e_expanded * batch_f_reconstructed) % self.field_size
                
                # 累加结果（对邻居列表的每个位置）
                batch_contribution = np.sum(batch_result, axis=0) % self.field_size
                result_accumulator = (result_accumulator + batch_contribution) % self.field_size
                
                # 清理缓存
                for local_idx in range(batch_size):
                    global_idx = batch_start + local_idx
                    state = all_computation_states[global_idx]
                    computation_id = state['computation_id']
                    if computation_id in self.mult_server.computation_cache:
                        del self.mult_server.computation_cache[computation_id]
            
            phase4_time = time.time() - phase4_start
            total_time = time.time() - start_time
            
            logger.info(f"查询完成:")
            logger.info(f"  阶段1 (VDPF): {phase1_time:.2f}秒")
            logger.info(f"  阶段2 (e/f): {phase2_time:.2f}秒")
            logger.info(f"  阶段3 (交换): {phase3_time:.2f}秒")
            logger.info(f"  阶段4 (重构): {phase4_time:.2f}秒")
            logger.info(f"  总计: {total_time:.2f}秒")
            
            # 输出网络传输统计
            network_stats = self._get_network_stats_summary()
            if network_stats:
                logger.info(f"网络传输统计:")
                logger.info(f"  发送数据: {network_stats['total_data_sent_mb']:.2f} MB")
                logger.info(f"  接收数据: {network_stats['total_data_received_mb']:.2f} MB")
                logger.info(f"  平均传输速度: {network_stats['avg_speed_mb_per_sec']:.2f} MB/s ({network_stats['avg_speed_mbps']:.2f} Mbps)")
                logger.info(f"  传输次数: {network_stats['total_transfers']}")
            
            logger.info("跳过文件同步屏障（分布式环境）")
            
            # 清理文件
            self._cleanup_query_files(query_id)
            
            # 返回结果 - 邻居列表索引
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
    
    def _cleanup_query_files(self, query_id: str):
        """清理查询相关文件和交换数据"""
        # 清理文件系统（如果有）
        for filename in os.listdir(self.exchange_dir):
            if query_id in filename:
                try:
                    os.remove(os.path.join(self.exchange_dir, filename))
                except:
                    pass
        
        # 清理内存中的交换数据
        if query_id in self.exchange_storage:
            del self.exchange_storage[query_id]
            logger.info(f"清理查询 {query_id} 的交换数据")
    
    def _get_network_stats_summary(self) -> Dict:
        """获取网络传输统计摘要"""
        if not self.network_stats['transfer_details']:
            return {}
        
        transfers = self.network_stats['transfer_details']
        total_data_mb = sum(t['data_size_mb'] for t in transfers)
        avg_speed_mbps = sum(t['speed_mbps'] for t in transfers) / len(transfers)
        avg_speed_mb_per_sec = sum(t['speed_mb_per_sec'] for t in transfers) / len(transfers)
        
        return {
            'total_transfers': len(transfers),
            'total_data_sent_mb': total_data_mb,
            'total_data_received_mb': self.network_stats['total_bytes_received'] / 1024 / 1024,
            'avg_speed_mbps': avg_speed_mbps,
            'avg_speed_mb_per_sec': avg_speed_mb_per_sec,
            'transfer_details': transfers[-5:]  # 最近5次传输详情
        }
    
    def _get_status(self) -> Dict:
        """获取服务器状态"""
        return {
            'status': 'success',
            'server_id': self.server_id,
            'mode': 'distributed-neighbor',
            'host': self.host,
            'port': self.port,
            'dataset': self.dataset,
            'vdpf_processes': self.vdpf_processes,
            'data_loaded': {
                'neighbors': self.neighbor_shares.shape,
                'k_neighbors': self.k_neighbors
            },
            'triples_available': self.mult_server.triple_array.shape[0] - self.mult_server.used_count if self.mult_server.triple_array is not None else 0,
            'network_stats': self._get_network_stats_summary()
        }


def main():
    """主函数"""
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='分布式邻居列表查询服务器')
    parser.add_argument('--server-id', type=int, required=True, choices=[1, 2, 3],
                        help='服务器ID (1, 2, 或 3)')
    parser.add_argument('--dataset', type=str, default='siftsmall', 
                        choices=['laion', 'siftsmall', 'tripclick', 'ms_marco', 'nfcorpus'],
                        help='数据集名称 (默认: siftsmall)')
    parser.add_argument('--vdpf-processes', type=int, default=4,
                        help='VDPF评估进程数 (默认: 4)')
    
    args = parser.parse_args()
    
    server = DistributedNeighborServer(args.server_id, args.dataset, args.vdpf_processes)
    server.start()


if __name__ == "__main__":
    main()