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

# Add project path
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

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [NL-Server] %(message)s')
logger = logging.getLogger(__name__)

# Global function for process pool
def warmup_process(process_id):
    """Warm up process, load necessary modules"""
    import time
    import sys
    sys.path.append('~/trident/query-opti')
    from dpf_wrapper_optimized import OptimizedVDPFVectorWrapper
    from binary_serializer import BinaryKeySerializer
    return process_id

def evaluate_batch_range_process(args):
    """Process pool worker: Evaluate specified batch range of VDPF[CN]"""
    process_id, start_batch, end_batch, cache_batch_size, num_nodes, serialized_key, server_id, shm_name, shape, dtype, dataset_name = args
    
    # CPU affinity binding
    try:
        import sys
        sys.path.append('~/trident/query-opti')
        from cpu_affinity_optimizer import set_process_affinity
        total_cores = cpu_count()
        
        if process_id < total_cores:
            import os
            pid = os.getpid()
            os.sched_setaffinity(pid, {process_id})
            print(f"[NL-Server {server_id}, Process {process_id}] bind[CN] {process_id}")
        else:
            core_id = process_id % total_cores
            import os
            pid = os.getpid()
            os.sched_setaffinity(pid, {core_id})
            print(f"[NL-Server {server_id}, Process {process_id}] bind[CN] {core_id}")
    except Exception as e:
        print(f"[Process {process_id}] CPU binding failed: {e}")
    
    process_total_start = time.time()
    
    # Calculate actualprocess[CN]
    actual_lists = 0
    for batch_idx in range(start_batch, end_batch):
        batch_start = batch_idx * cache_batch_size
        batch_end = min(batch_start + cache_batch_size, num_nodes)
        actual_lists += (batch_end - batch_start)
    
    # VDPFinstancecreate
    from dpf_wrapper_optimized import OptimizedVDPFVectorWrapper
    dpf_wrapper = OptimizedVDPFVectorWrapper(dataset_name=dataset_name)
    
    # [CN]
    if isinstance(serialized_key, bytes):
        key = BinaryKeySerializer.deserialize_vdpf23_key(serialized_key)
    else:
        key = dpf_wrapper._deserialize_key(serialized_key)
    
    # connect[CN]（[CN]）
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    neighbor_shares = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    
    local_selector_shares = {}
    local_neighbor_shares = {}
    
    # VDPF[CN] - [CN]
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
    
    # [CN]connect
    existing_shm.close()
    
    return {
        'selector_shares': local_selector_shares,
        'neighbor_shares': local_neighbor_shares
    }


class DistributedNeighborServer:
    """[CN]"""
    
    def __init__(self, server_id: int, dataset: str = "siftsmall", vdpf_processes: int = 32):
        self.server_id = server_id
        self.dataset = dataset
        self.config = get_config(dataset)
        self.field_size = self.config.prime
        self.mpc = MPC23SSS(self.config)
        
        # [CN] - [CN]
        self.host = "0.0.0.0"  # [CN]connect
        self.port = 9000 + server_id  # [CN]9001-9003[CN]，[CN]
        
        # initialize[CN]
        self.dpf_wrapper = OptimizedVDPFVectorWrapper(dataset_name=dataset)
        self.mult_server = NumpyMultiplicationServer(server_id, self.config)
        
        # [CN]
        self._load_neighbor_data()
        
        # [CN]
        self.exchange_dir = "/tmp/mpc_nl_exchange"
        os.makedirs(self.exchange_dir, exist_ok=True)
        
        # [CN]
        self._cleanup_old_files()
        
        # [CN]
        self.cache_batch_size = max(100, 1000 // max(vdpf_processes // 4, 1))
        self.vdpf_processes = vdpf_processes
        self.worker_threads = 8
        logger.info(f"[CN]process[CN]: {self.cache_batch_size}, VDPF[CN]: {self.vdpf_processes}")
        
        # create[CN]
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.worker_threads,
            thread_name_prefix=f"NLServer{server_id}-General"
        )
        
        # create[CN]
        self.process_pool = Pool(processes=self.vdpf_processes)
        
        # [CN]
        self.exchange_storage = {}  # {query_id: {'e_shares': ..., 'f_shares': ...}}
        
        # [CN]connect[CN]
        self.server_connections = {}  # {server_id: socket}
        self.server_config = SERVER_TO_SERVER  # [CN]IP
        
        # [CN]
        self.network_stats = {
            'total_bytes_sent': 0,
            'total_bytes_received': 0,
            'transfer_count': 0,
            'transfer_details': []
        }
        
        logger.info(f"[CN]initialize[CN]")
        logger.info(f"[CN]: {self.host}:{self.port}")
        logger.info(f"Dataset: {self.dataset}")
        logger.info(f"VDPF[CN]: {self.vdpf_processes}")
        
        # Warm up process[CN]
        self._warmup_process_pool()
        
        # [CN]connect
        self.connections_established = False
        
    def _cleanup_old_files(self):
        """[CN]"""
        try:
            for filename in os.listdir(self.exchange_dir):
                if f"nl_server_{self.server_id}_" in filename:
                    try:
                        os.remove(os.path.join(self.exchange_dir, filename))
                    except:
                        pass
            logger.info("[CN]")
        except:
            pass
    
    def _warmup_process_pool(self):
        """Warm up process[CN]"""
        logger.info("Warm up process[CN]...")
        warmup_start = time.time()
        results = self.process_pool.map(warmup_process, range(self.vdpf_processes))
        warmup_time = time.time() - warmup_start
        logger.info(f"[CN]，[CN] {warmup_time:.2f}[CN]")
    
    def _load_neighbor_data(self):
        """[CN]"""
        logger.info(f"[CN]{self.dataset}[CN]...")
        
        self.data_dir = f"~/trident/dataset/{self.dataset}/server_{self.server_id}"
        
        # [CN]
        self.neighbors_path = os.path.join(self.data_dir, "neighbors_shares.npy")
        self.neighbor_shares = np.load(self.neighbors_path)
        logger.info(f"[CN]: {self.neighbor_shares.shape}")
        logger.info(f"[CN]: {self.neighbor_shares.nbytes / 1024 / 1024:.1f}MB")
        logger.info(f"[CN]: {self.neighbor_shares.dtype}")
        
        # [CN]K（[CN]）
        self.num_nodes = self.neighbor_shares.shape[0]
        self.k_neighbors = self.neighbor_shares.shape[1] if len(self.neighbor_shares.shape) > 1 else 1
        logger.info(f"[CN]: {self.num_nodes}, [CN]: {self.k_neighbors}")
        
        logger.info(f"[CN]: {self.mult_server.triple_array.shape[0] - self.mult_server.used_count if self.mult_server.triple_array is not None else 0}")
    
    def start(self):
        """start[CN]"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # [CN]socket[CN]
        server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        
        try:
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)
            logger.info(f"[CN]start[CN]，[CN] {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"[CN]bind[CN] {self.host}:{self.port}: {e}")
            return
        
        try:
            while True:
                client_socket, address = server_socket.accept()
                logger.info(f"[CN]connect[CN] {address}")
                
                # [CN]socket[CN]
                client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address)
                )
                client_thread.daemon = True
                client_thread.start()
                
        except KeyboardInterrupt:
            logger.info("receive[CN]，[CN]...")
        except Exception as e:
            logger.error(f"[CN]: {e}")
        finally:
            server_socket.close()
            self._cleanup_resources()
            logger.info("[CN]")
    
    def _cleanup_resources(self):
        """[CN]"""
        self._cleanup_old_files()
        self.executor.shutdown(wait=True)
        if hasattr(self, 'process_pool'):
            logger.info("[CN]...")
            self.process_pool.close()
            self.process_pool.join()
    
    def _handle_client(self, client_socket: socket.socket, address):
        """process[CN]"""
        try:
            while True:
                # [CN]
                length_bytes = client_socket.recv(4)
                if not length_bytes:
                    break
                
                length = int.from_bytes(length_bytes, 'big')
                
                # [CN]
                data = b''
                while len(data) < length:
                    chunk = client_socket.recv(min(length - len(data), 4096))
                    if not chunk:
                        break
                    data += chunk
                
                if len(data) < length:
                    logger.warning(f"[CN] {address} receive[CN]")
                    break
                
                # [CN]
                try:
                    # [CN]
                    first_byte = data[0] if data else -1
                    logger.info(f"receive[CN]: {first_byte}, [CN]: {len(data)}")
                    logger.info(f"[CN]: NODE_VECTOR={BinaryProtocol.CMD_QUERY_NODE_VECTOR}, STATUS={BinaryProtocol.CMD_GET_STATUS}, NEIGHBOR_LIST={BinaryProtocol.CMD_QUERY_NEIGHBOR_LIST}")
                    
                    if first_byte in [BinaryProtocol.CMD_QUERY_NODE_VECTOR, BinaryProtocol.CMD_GET_STATUS, BinaryProtocol.CMD_QUERY_NEIGHBOR_LIST]:
                        request = BinaryProtocol.decode_request(data)
                        logger.info(f"[CN]: {request.get('command', 'unknown')}")
                    else:
                        # [CN]JSON[CN]
                        logger.info(f"[CN]JSON[CN]，[CN]20[CN]: {data[:20]}")
                        request = json.loads(data.decode())
                        logger.info(f"[CN]JSON[CN]: {request.get('command', 'unknown')}")
                except Exception as e:
                    logger.error(f"[CN]: {e}")
                    continue
                
                # [CN]process[CN]
                if request.get('command') == 'binary_exchange_data':
                    receive_start_time = time.time()
                    from_server = request.get('from_server')
                    query_id = request.get('query_id')
                    
                    logger.info(f"receive[CN] {from_server} [CN] ([CN]: {query_id})")
                    logger.info(f"  [CN]: e_shares={request.get('e_shares_shape')}, f_shares={request.get('f_shares_shape')}")
                    
                    # receivee_shares - [CN]MSG_WAITALL[CN]
                    e_len_bytes = client_socket.recv(4, socket.MSG_WAITALL)
                    if len(e_len_bytes) != 4:
                        logger.error(f"receivee_shares[CN]: {len(e_len_bytes)} bytes")
                        continue
                    
                    e_len = int.from_bytes(e_len_bytes, 'big')
                    logger.info(f"  receive e_shares: {e_len:,} bytes")
                    
                    e_data = client_socket.recv(e_len, socket.MSG_WAITALL)
                    if len(e_data) != e_len:
                        logger.error(f"e_sharesreceive[CN]: {len(e_data)}/{e_len} bytes")
                        continue
                    
                    # receivef_shares - [CN]MSG_WAITALL[CN]
                    f_len_bytes = client_socket.recv(4, socket.MSG_WAITALL)
                    if len(f_len_bytes) != 4:
                        logger.error(f"receivef_shares[CN]: {len(f_len_bytes)} bytes")
                        continue
                    
                    f_len = int.from_bytes(f_len_bytes, 'big')
                    logger.info(f"  receive f_shares: {f_len:,} bytes")
                    
                    f_data = client_socket.recv(f_len, socket.MSG_WAITALL)
                    if len(f_data) != f_len:
                        logger.error(f"f_sharesreceive[CN]: {len(f_data)}/{f_len} bytes")
                        continue
                    
                    total_received_size = len(data) + len(e_data) + len(f_data) + 8  # +8 for length headers
                    logger.info(f"  [CN]receive: {total_received_size:,} bytes ({total_received_size/1024/1024:.2f} MB)")
                    
                    # [CN]
                    e_shares = np.frombuffer(e_data, dtype=np.uint64)
                    f_shape = tuple(request['f_shares_shape'])
                    f_shares = np.frombuffer(f_data, dtype=np.uint64).reshape(f_shape)
                    
                    # [CN]
                    if query_id not in self.exchange_storage:
                        self.exchange_storage[query_id] = {}
                    
                    self.exchange_storage[query_id][f'e_shares_{from_server}'] = e_shares
                    self.exchange_storage[query_id][f'f_shares_{from_server}'] = f_shares
                    
                    # calculatereceive[CN]
                    receive_duration = time.time() - receive_start_time
                    receive_speed = total_received_size / receive_duration if receive_duration > 0 else 0
                    
                    # [CN]
                    self.network_stats['total_bytes_received'] += total_received_size
                    
                    logger.info(f"[CN]receive[CN]:")
                    logger.info(f"  [CN]: [CN] {from_server}")
                    logger.info(f"  [CN]: {receive_duration:.3f}[CN]")
                    logger.info(f"  [CN]: {receive_speed/1024/1024:.2f} MB/s ({(receive_speed*8)/1024/1024:.2f} Mbps)")
                    
                    # send[CN]
                    response = {'status': 'success'}
                    response_data = json.dumps(response).encode()
                    client_socket.sendall(len(response_data).to_bytes(4, 'big'))
                    client_socket.sendall(response_data)
                    continue
                
                # process[CN]
                response = self._process_request(request)
                
                # send[CN]
                response_data = BinaryProtocol.encode_response(response)
                client_socket.sendall(response_data)
                
        except ConnectionResetError:
            logger.info(f"[CN] {address} [CN]connect")
        except Exception as e:
            logger.error(f"process[CN] {address} [CN]: {e}")
            import traceback
            traceback.print_exc()
        finally:
            client_socket.close()
    
    def _process_request(self, request: Dict) -> Dict:
        """process[CN]"""
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
            return {'status': 'error', 'message': f'[CN]: {command}'}
    
    def _establish_persistent_connections(self):
        """[CN]connect"""
        logger.info("[CN]connect[CN]...")
        
        for server_id, server_info in self.server_config.items():
            if server_id == self.server_id or server_id in self.server_connections:
                continue
            
            # [CN]connect[CN]（9001-9003）
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024 * 1024)
                sock.settimeout(1)
                # [CN]
                sock.connect((server_info['host'], 9000 + server_id))
                self.server_connections[server_id] = sock
                logger.info(f"[CN]connect[CN] {server_id}")
            except Exception as e:
                logger.debug(f"[CN] {server_id} [CN]: {e}")
    
    def _send_binary_exchange_data(self, target_server_id: int, query_id: str, e_shares: np.ndarray, f_shares: np.ndarray) -> bool:
        """send[CN]"""
        send_start_time = time.time()
        
        # [CN]connect
        if not self.connections_established:
            logger.info("[CN]，[CN]connect...")
            self._establish_persistent_connections()
            self.connections_established = True
        
        # [CN]connect[CN]
        if target_server_id not in self.server_connections:
            logger.warning(f"[CN] {target_server_id} [CN]connect，[CN]connect...")
            # [CN]connect
            try:
                server_info = self.server_config[target_server_id]
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024 * 1024)
                sock.settimeout(30)
                # connect[CN]
                sock.connect((server_info['host'], 9000 + target_server_id))
                self.server_connections[target_server_id] = sock
                logger.info(f"[CN]connect[CN] {target_server_id}")
            except Exception as e:
                logger.error(f"[CN]connect[CN] {target_server_id}: {e}")
                return False
        else:
            logger.debug(f"[CN]connect[CN] {target_server_id}")
        
        try:
            sock = self.server_connections[target_server_id]
            
            # [CN]
            metadata = {
                'command': 'binary_exchange_data',
                'query_id': query_id,
                'from_server': self.server_id,
                'e_shares_shape': e_shares.shape,
                'f_shares_shape': f_shares.shape
            }
            
            # calculate[CN]
            e_bytes = e_shares.tobytes()
            f_bytes = f_shares.tobytes()
            metadata_bytes = json.dumps(metadata).encode()
            
            # send[CN]
            sock.sendall(len(metadata_bytes).to_bytes(4, 'big'))
            sock.sendall(metadata_bytes)
            
            total_data_size = len(metadata_bytes) + len(e_bytes) + len(f_bytes) + 12  # +12 for length headers
            
            logger.info(f"send[CN] {target_server_id}:")
            logger.info(f"  e_shares: {e_shares.shape}, {len(e_bytes):,} bytes")
            logger.info(f"  f_shares: {f_shares.shape}, {len(f_bytes):,} bytes")
            logger.info(f"  [CN]: {len(metadata_bytes):,} bytes")
            logger.info(f"  [CN]: {total_data_size:,} bytes ({total_data_size/1024/1024:.2f} MB)")
            
            # sende_shares
            sock.sendall(len(e_bytes).to_bytes(4, 'big'))
            sock.sendall(e_bytes)
            
            # sendf_shares
            sock.sendall(len(f_bytes).to_bytes(4, 'big'))
            sock.sendall(f_bytes)
            
            # receive[CN]
            response_len = int.from_bytes(sock.recv(4), 'big')
            response_data = sock.recv(response_len)
            response = json.loads(response_data.decode())
            
            # calculate[CN]
            send_duration = time.time() - send_start_time
            transfer_speed = total_data_size / send_duration if send_duration > 0 else 0
            
            # [CN]
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
            
            logger.info(f"[CN]:")
            logger.info(f"  [CN]: {send_duration:.3f}[CN]")
            logger.info(f"  [CN]: {transfer_speed/1024/1024:.2f} MB/s ({(transfer_speed*8)/1024/1024:.2f} Mbps)")
            
            return response.get('status') == 'success'
            
        except Exception as e:
            logger.error(f"send[CN] {target_server_id} [CN]: {e}")
            # [CN]connect
            if target_server_id in self.server_connections:
                try:
                    self.server_connections[target_server_id].close()
                except:
                    pass
                del self.server_connections[target_server_id]
            return False
    
    def _exchange_data_with_servers(self, query_id: str, e_shares: np.ndarray, f_shares: np.ndarray) -> tuple:
        """[CN]"""
        send_start = time.time()
        
        # [CN]send[CN]
        def send_to_server_async(server_id):
            if server_id == self.server_id:
                return None
            start = time.time()
            success = self._send_binary_exchange_data(server_id, query_id, e_shares, f_shares)
            elapsed = time.time() - start
            if success:
                logger.info(f"[CN] {server_id} send[CN] ([CN]: {elapsed:.3f}[CN])")
            else:
                logger.error(f"[CN] {server_id} send[CN]")
            return server_id, success
        
        # [CN]send
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(send_to_server_async, sid) for sid in [1, 2, 3] if sid != self.server_id]
            concurrent.futures.wait(futures)
        
        send_time = time.time() - send_start
        logger.info(f"send[CN]: {send_time:.3f}[CN]")
        
        # [CN]receive[CN]
        max_wait = 30  # [CN]30[CN]
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            # [CN]
            received_all = True
            for server_id in [1, 2, 3]:
                if server_id == self.server_id:
                    continue
                
                if query_id not in self.exchange_storage or \
                   f'e_shares_{server_id}' not in self.exchange_storage[query_id]:
                    received_all = False
                    break
            
            if received_all:
                # [CN]
                all_e_from_others = {}
                all_f_from_others = {}
                
                for server_id in [1, 2, 3]:
                    if server_id == self.server_id:
                        continue
                    
                    all_e_from_others[server_id] = self.exchange_storage[query_id][f'e_shares_{server_id}']
                    all_f_from_others[server_id] = self.exchange_storage[query_id][f'f_shares_{server_id}']
                
                return all_e_from_others, all_f_from_others
            
            time.sleep(0.1)
        
        logger.error(f"[CN] ([CN]: {query_id})")
        return {}, {}
    
    def _multiprocess_vdpf_evaluation(self, serialized_key, num_nodes, num_batches):
        """[CN]VDPF[CN] - [CN]"""
        # create[CN]
        shm = shared_memory.SharedMemory(create=True, size=self.neighbor_shares.nbytes)
        shared_array = np.ndarray(self.neighbor_shares.shape, dtype=self.neighbor_shares.dtype, buffer=shm.buf)
        shared_array[:] = self.neighbor_shares[:]
        
        # [CN]allocate
        nodes_per_process = num_nodes // self.vdpf_processes
        remaining_nodes = num_nodes % self.vdpf_processes
        
        # [CN]
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
        
        # [CN]
        results = self.process_pool.map(evaluate_batch_range_process, process_args)
        
        # [CN]
        all_selector_shares = {}
        all_neighbor_shares = {}
        
        for process_result in results:
            all_selector_shares.update(process_result['selector_shares'])
            all_neighbor_shares.update(process_result['neighbor_shares'])
        
        # [CN]
        shm.close()
        shm.unlink()
        
        return all_selector_shares, all_neighbor_shares
    
    def _handle_neighbor_list_query(self, request: Dict) -> Dict:
        """process[CN]"""
        try:
            serialized_key = request['dpf_key']
            query_id = request.get('query_id', 'unknown')
            
            logger.info(f"process[CN]，[CN]ID: {query_id}")
            
            # Deserialize keys
            key = self.dpf_wrapper._deserialize_key(serialized_key)
            
            # initialize
            start_time = time.time()
            num_nodes = len(self.neighbor_shares)
            result_accumulator = np.zeros(self.k_neighbors, dtype=np.int64)
            
            # [CN]1：[CN]VDPF[CN]
            logger.info(f"[CN]1: [CN]VDPF[CN] ({self.vdpf_processes} [CN])...")
            phase1_start = time.time()
            
            num_batches = (num_nodes + self.cache_batch_size - 1) // self.cache_batch_size
            
            all_selector_shares, all_neighbor_shares = self._multiprocess_vdpf_evaluation(
                serialized_key, num_nodes, num_batches)
            
            phase1_time = time.time() - phase1_start
            logger.info(f"[CN]1[CN]，[CN] {phase1_time:.2f}[CN]")
            
            # [CN]2：e/fcalculate
            logger.info("[CN]2: e/fcalculate...")
            phase2_start = time.time()
            
            all_e_shares = np.zeros(num_nodes, dtype=np.uint64)
            all_f_shares = np.zeros((num_nodes, self.k_neighbors), dtype=np.uint64)
            all_computation_states = {}
            
            logger.info(f"initialize e_shares: dtype={all_e_shares.dtype}, shape={all_e_shares.shape}")
            logger.info(f"initialize f_shares: dtype={all_f_shares.dtype}, shape={all_f_shares.shape}")
            
            for batch_idx in range(num_batches):
                batch_start = batch_idx * self.cache_batch_size
                batch_end = min(batch_start + self.cache_batch_size, num_nodes)
                batch_size = batch_end - batch_start
                
                # [CN]
                batch_triples = []
                for _ in range(batch_size):
                    batch_triples.append(self.mult_server.get_next_triple())
                
                # process[CN]
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
            logger.info(f"[CN]2[CN]，[CN] {phase2_time:.2f}[CN]")
            
            # [CN]3：[CN]
            logger.info("[CN]3: [CN]...")
            phase3_start = time.time()
            
            # [CN]
            all_e_from_others, all_f_from_others = self._exchange_data_with_servers(query_id, all_e_shares, all_f_shares)
            
            phase3_time = time.time() - phase3_start
            logger.info(f"[CN]3[CN]，[CN] {phase3_time:.2f}[CN]")
            
            # [CN]4：[CN]calculate
            logger.info("[CN]4: [CN]calculate...")
            phase4_start = time.time()
            
            # [CN]
            lagrange_1 = 2
            lagrange_2 = self.field_size - 1
            
            # [CN]
            for batch_idx in range(num_batches):
                batch_start = batch_idx * self.cache_batch_size
                batch_end = min(batch_start + self.cache_batch_size, num_nodes)
                batch_size = batch_end - batch_start
                
                # [CN]
                batch_e_shares_local = all_e_shares[batch_start:batch_end]
                batch_f_shares_local = all_f_shares[batch_start:batch_end, :]
                
                # [CN]
                e_shares_matrix = np.zeros((batch_size, 3), dtype=np.uint64)
                e_shares_matrix[:, self.server_id - 1] = batch_e_shares_local
                
                f_shares_matrix = np.zeros((batch_size, self.k_neighbors, 3), dtype=np.uint64)
                f_shares_matrix[:, :, self.server_id - 1] = batch_f_shares_local
                
                # [CN]
                for other_id, other_e_shares in all_e_from_others.items():
                    e_shares_matrix[:, other_id - 1] = other_e_shares[batch_start:batch_end]
                
                for other_id, other_f_shares in all_f_from_others.items():
                    f_shares_matrix[:, :, other_id - 1] = other_f_shares[batch_start:batch_end, :]
                
                # [CN]
                batch_e_reconstructed = (e_shares_matrix[:, 0] * lagrange_1 + 
                                       e_shares_matrix[:, 1] * lagrange_2) % self.field_size
                
                batch_f_reconstructed = (f_shares_matrix[:, :, 0] * lagrange_1 + 
                                       f_shares_matrix[:, :, 1] * lagrange_2) % self.field_size
                
                # [CN]
                batch_triples = []
                for local_idx in range(batch_size):
                    global_idx = batch_start + local_idx
                    state = all_computation_states[global_idx]
                    a, b, c = state['triple']
                    batch_triples.append((a, b, c))
                
                batch_a = np.array([t[0] for t in batch_triples], dtype=np.uint64)
                batch_b = np.array([t[1] for t in batch_triples], dtype=np.uint64)
                batch_c = np.array([t[2] for t in batch_triples], dtype=np.uint64)
                
                # calculate[CN] - [CN]calculate
                batch_e_expanded = batch_e_reconstructed[:, np.newaxis]
                batch_a_expanded = batch_a[:, np.newaxis]
                batch_b_expanded = batch_b[:, np.newaxis]
                batch_c_expanded = batch_c[:, np.newaxis]
                
                batch_result = batch_c_expanded
                batch_result = (batch_result + batch_e_expanded * batch_b_expanded) % self.field_size
                batch_result = (batch_result + batch_f_reconstructed * batch_a_expanded) % self.field_size
                batch_result = (batch_result + batch_e_expanded * batch_f_reconstructed) % self.field_size
                
                # [CN]（[CN]）
                batch_contribution = np.sum(batch_result, axis=0) % self.field_size
                result_accumulator = (result_accumulator + batch_contribution) % self.field_size
                
                # [CN]
                for local_idx in range(batch_size):
                    global_idx = batch_start + local_idx
                    state = all_computation_states[global_idx]
                    computation_id = state['computation_id']
                    if computation_id in self.mult_server.computation_cache:
                        del self.mult_server.computation_cache[computation_id]
            
            phase4_time = time.time() - phase4_start
            total_time = time.time() - start_time
            
            logger.info(f"[CN]:")
            logger.info(f"  [CN]1 (VDPF): {phase1_time:.2f}[CN]")
            logger.info(f"  [CN]2 (e/f): {phase2_time:.2f}[CN]")
            logger.info(f"  [CN]3 ([CN]): {phase3_time:.2f}[CN]")
            logger.info(f"  [CN]4 ([CN]): {phase4_time:.2f}[CN]")
            logger.info(f"  [CN]: {total_time:.2f}[CN]")
            
            # [CN]
            network_stats = self._get_network_stats_summary()
            if network_stats:
                logger.info(f"[CN]:")
                logger.info(f"  send[CN]: {network_stats['total_data_sent_mb']:.2f} MB")
                logger.info(f"  receive[CN]: {network_stats['total_data_received_mb']:.2f} MB")
                logger.info(f"  [CN]: {network_stats['avg_speed_mb_per_sec']:.2f} MB/s ({network_stats['avg_speed_mbps']:.2f} Mbps)")
                logger.info(f"  [CN]: {network_stats['total_transfers']}")
            
            logger.info("[CN]（[CN]）")
            
            # [CN]
            self._cleanup_query_files(query_id)
            
            # return[CN] - [CN]
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
            logger.error(f"[CN]process[CN]: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}
    
    def _cleanup_query_files(self, query_id: str):
        """[CN]"""
        # [CN]（[CN]）
        for filename in os.listdir(self.exchange_dir):
            if query_id in filename:
                try:
                    os.remove(os.path.join(self.exchange_dir, filename))
                except:
                    pass
        
        # [CN]
        if query_id in self.exchange_storage:
            del self.exchange_storage[query_id]
            logger.info(f"[CN] {query_id} [CN]")
    
    def _get_network_stats_summary(self) -> Dict:
        """[CN]"""
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
            'transfer_details': transfers[-5:]  # [CN]5[CN]
        }
    
    def _get_status(self) -> Dict:
        """[CN]"""
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
    """[CN]"""
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='[CN]')
    parser.add_argument('--server-id', type=int, required=True, choices=[1, 2, 3],
                        help='[CN]ID (1, 2, [CN] 3)')
    parser.add_argument('--dataset', type=str, default='siftsmall', 
                        choices=['laion', 'siftsmall', 'tripclick', 'ms_marco', 'nfcorpus'],
                        help='Dataset[CN] ([CN]: siftsmall)')
    parser.add_argument('--vdpf-processes', type=int, default=4,
                        help='VDPF[CN] ([CN]: 4)')
    
    args = parser.parse_args()
    
    server = DistributedNeighborServer(args.server_id, args.dataset, args.vdpf_processes)
    server.start()


if __name__ == "__main__":
    main()