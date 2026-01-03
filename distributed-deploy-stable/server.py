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

# Add project paths
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
from config import SERVERS

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [Server] %(message)s')
logger = logging.getLogger(__name__)

# Global function for process pool calls
def warmup_process(process_id):
    """Warm up process and load necessary modules"""
    import time
    import sys
    sys.path.append('~/trident/query-opti')
    from dpf_wrapper_optimized import OptimizedVDPFVectorWrapper
    from binary_serializer import BinaryKeySerializer
    return process_id

def evaluate_batch_range_process(args):
    """Process pool worker function: evaluate VDPF for specified batch range"""
    process_id, start_batch, end_batch, cache_batch_size, num_nodes, serialized_key, server_id, shm_name, shape, dtype, dataset_name = args

    # CPU affinity binding
    try:
        import sys
        sys.path.append('~/trident/query-opti')
        from cpu_affinity_optimizer import set_process_affinity
        total_cores = cpu_count()

        # Each server has its own 64 cores, directly use process_id as core ID
        if process_id < total_cores:
            import os
            pid = os.getpid()
            os.sched_setaffinity(pid, {process_id})
            print(f"[Server {server_id}, Process {process_id}] Bound to core {process_id}")
        else:
            # If process count exceeds core count, allocate in round-robin
            core_id = process_id % total_cores
            import os
            pid = os.getpid()
            os.sched_setaffinity(pid, {core_id})
            print(f"[Server {server_id}, Process {process_id}] Bound to core {core_id}")
    except Exception as e:
        print(f"[Process {process_id}] CPU binding failed: {e}")

    process_total_start = time.time()

    # Calculate actual number of nodes to process
    actual_nodes = 0
    for batch_idx in range(start_batch, end_batch):
        batch_start = batch_idx * cache_batch_size
        batch_end = min(batch_start + cache_batch_size, num_nodes)
        actual_nodes += (batch_end - batch_start)

    # Create VDPF instance
    from dpf_wrapper_optimized import OptimizedVDPFVectorWrapper
    dpf_wrapper = OptimizedVDPFVectorWrapper(dataset_name=dataset_name)

    # Deserialize key
    if isinstance(serialized_key, bytes):
        key = BinaryKeySerializer.deserialize_vdpf23_key(serialized_key)
    else:
        key = dpf_wrapper._deserialize_key(serialized_key)

    # Connect to shared memory
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    node_shares = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

    local_selector_shares = {}
    local_vector_shares = {}

    # VDPF evaluation
    for batch_idx in range(start_batch, end_batch):
        batch_start = batch_idx * cache_batch_size
        batch_end = min(batch_start + cache_batch_size, num_nodes)
        batch_size = batch_end - batch_start

        batch_data = node_shares[batch_start:batch_end].copy()
        batch_results = dpf_wrapper.eval_batch(key, batch_start, batch_end, server_id)

        for local_idx in range(batch_size):
            global_idx = batch_start + local_idx
            local_selector_shares[global_idx] = batch_results[global_idx]
            local_vector_shares[global_idx] = batch_data[local_idx]

    # Close shared memory connection
    existing_shm.close()

    return {
        'selector_shares': local_selector_shares,
        'vector_shares': local_vector_shares
    }


class DistributedServer:
    """Distributed server for real network environment"""

    def __init__(self, server_id: int, dataset: str = "siftsmall", vdpf_processes: int = 32):
        self.server_id = server_id
        self.dataset = dataset
        self.config = get_config(dataset)
        self.field_size = self.config.prime
        self.mpc = MPC23SSS(self.config)

        # Network configuration - listen on all interfaces
        self.host = "0.0.0.0"  # Accept all external connections
        self.port = 8000 + server_id

        # Initialize components
        self.dpf_wrapper = OptimizedVDPFVectorWrapper(dataset_name=dataset)
        self.mult_server = NumpyMultiplicationServer(server_id, self.config)

        # Load data
        self._load_data()

        # Exchange directory
        self.exchange_dir = "/tmp/mpc_exchange"
        os.makedirs(self.exchange_dir, exist_ok=True)

        # Clean up old files
        self._cleanup_old_files()

        # Multi-process optimization parameters
        # Dynamically adjust batch size based on process count, each process handles fewer batches to improve parallelism
        self.cache_batch_size = max(100, 1000 // max(vdpf_processes // 4, 1))
        self.vdpf_processes = vdpf_processes
        self.worker_threads = 8  # Increase number of worker threads
        logger.info(f"Using batch size: {self.cache_batch_size}, VDPF processes: {self.vdpf_processes}")

        # Create thread pool
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.worker_threads,
            thread_name_prefix=f"Server{server_id}-General"
        )

        # Create process pool
        self.process_pool = Pool(processes=self.vdpf_processes)

        # Data exchange storage
        self.exchange_storage = {}  # {query_id: {'e_shares': ..., 'f_shares': ...}}

        # Inter-server connection management
        self.server_connections = {}  # {server_id: socket}
        self.server_config = SERVERS  # Server configuration imported from config

        logger.info(f"Distributed server initialization complete")
        logger.info(f"Listening on: {self.host}:{self.port}")
        logger.info(f"Dataset: {self.dataset}")
        logger.info(f"VDPF processes: {self.vdpf_processes}")

        # Warm up process pool
        self._warmup_process_pool()

        # Lazy connection establishment
        self.connections_established = False
        
    def _cleanup_old_files(self):
        """Clean up old exchange files"""
        try:
            for filename in os.listdir(self.exchange_dir):
                if f"server_{self.server_id}_" in filename:
                    try:
                        os.remove(os.path.join(self.exchange_dir, filename))
                    except:
                        pass
            logger.info("Cleaned up old synchronization files")
        except:
            pass

    def _warmup_process_pool(self):
        """Warm up process pool"""
        logger.info("Warming up process pool...")
        warmup_start = time.time()
        results = self.process_pool.map(warmup_process, range(self.vdpf_processes))
        warmup_time = time.time() - warmup_start
        logger.info(f"Process pool warm up complete, elapsed {warmup_time:.2f}s")

    def _load_data(self):
        """Load vector-level secret shared data"""
        logger.info(f"Loading {self.dataset} data...")

        self.data_dir = f"~/trident/dataset/{self.dataset}/server_{self.server_id}"

        # Load node vector shares
        self.nodes_path = os.path.join(self.data_dir, "nodes_shares.npy")
        self.node_shares = np.load(self.nodes_path)
        logger.info(f"Node data: {self.node_shares.shape}")
        logger.info(f"Data size: {self.node_shares.nbytes / 1024 / 1024:.1f}MB")

        logger.info(f"Available triples: {self.mult_server.triple_array.shape[0] - self.mult_server.used_count if self.mult_server.triple_array is not None else 0}")
    
    def start(self):
        """Start server"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Set socket options to improve network performance
        server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        try:
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)
            logger.info(f"Server started successfully, listening on {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Unable to bind to {self.host}:{self.port}: {e}")
            return

        try:
            while True:
                client_socket, address = server_socket.accept()
                logger.info(f"Accepted connection from {address}")

                # Set client socket options
                client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address)
                )
                client_thread.daemon = True
                client_thread.start()

        except KeyboardInterrupt:
            logger.info("Received interrupt signal, closing server...")
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            server_socket.close()
            self._cleanup_resources()
            logger.info("Server closed")
    
    def _cleanup_resources(self):
        """Clean up resources"""
        self._cleanup_old_files()
        self.executor.shutdown(wait=True)
        if hasattr(self, 'process_pool'):
            logger.info("Closing process pool...")
            self.process_pool.close()
            self.process_pool.join()

    def _handle_client(self, client_socket: socket.socket, address):
        """Handle client requests"""
        try:
            while True:
                # Read request length
                length_bytes = client_socket.recv(4)
                if not length_bytes:
                    break

                length = int.from_bytes(length_bytes, 'big')

                # Read request data
                data = b''
                while len(data) < length:
                    chunk = client_socket.recv(min(length - len(data), 4096))
                    if not chunk:
                        break
                    data += chunk

                if len(data) < length:
                    logger.warning(f"Received incomplete data from {address}")
                    break

                # Parse request
                try:
                    # Check if it's binary protocol
                    if data[0] in [BinaryProtocol.CMD_QUERY_NODE_VECTOR, BinaryProtocol.CMD_GET_STATUS]:
                        request = BinaryProtocol.decode_request(data)
                        logger.info(f"Received binary request: {request.get('command', 'unknown')}")
                    else:
                        request = json.loads(data.decode())
                        logger.info(f"Received JSON request: {request.get('command', 'unknown')}")
                except Exception as e:
                    logger.error(f"Failed to parse request: {e}")
                    continue
                
                # Special handling for binary data exchange
                if request.get('command') == 'binary_exchange_data':
                    # Receive e_shares
                    e_len_bytes = client_socket.recv(4)
                    e_len = int.from_bytes(e_len_bytes, 'big')
                    e_data = b''
                    while len(e_data) < e_len:
                        chunk = client_socket.recv(min(e_len - len(e_data), 65536))
                        if not chunk:
                            break
                        e_data += chunk

                    # Receive f_shares
                    f_len_bytes = client_socket.recv(4)
                    f_len = int.from_bytes(f_len_bytes, 'big')
                    f_data = b''
                    while len(f_data) < f_len:
                        chunk = client_socket.recv(min(f_len - len(f_data), 65536))
                        if not chunk:
                            break
                        f_data += chunk

                    # Reconstruct arrays
                    e_shares = np.frombuffer(e_data, dtype=np.uint64)
                    f_shape = tuple(request['f_shares_shape'])
                    f_shares = np.frombuffer(f_data, dtype=np.uint64).reshape(f_shape)

                    # Store data
                    query_id = request['query_id']
                    from_server = request['from_server']

                    if query_id not in self.exchange_storage:
                        self.exchange_storage[query_id] = {}

                    self.exchange_storage[query_id][f'e_shares_{from_server}'] = e_shares
                    self.exchange_storage[query_id][f'f_shares_{from_server}'] = f_shares

                    logger.info(f"Received binary data from server {from_server} (query: {query_id})")

                    # Send acknowledgment response
                    response = {'status': 'success'}
                    response_data = json.dumps(response).encode()
                    client_socket.sendall(len(response_data).to_bytes(4, 'big'))
                    client_socket.sendall(response_data)
                    continue

                # Handle other requests
                response = self._process_request(request)

                # Send response
                response_data = BinaryProtocol.encode_response(response)
                client_socket.sendall(response_data)

        except ConnectionResetError:
            logger.info(f"Client {address} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {address}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            client_socket.close()
    
    def _process_request(self, request: Dict) -> Dict:
        """Process request"""
        command = request.get('command')

        if command == 'query_node_vector':
            return self._handle_vector_node_query(request)
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
            return {'status': 'error', 'message': f'Unknown command: {command}'}
    
    def _establish_persistent_connections(self):
        """Establish persistent connections to other servers"""
        logger.info("Establishing persistent connections to other servers...")

        for server_id, server_info in self.server_config.items():
            if server_id == self.server_id or server_id in self.server_connections:
                continue

            # Try to establish connection
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # Disable Nagle's algorithm
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)  # Enable keepalive
                # Increase socket buffer size
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)  # 16MB receive buffer
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024 * 1024)  # 16MB send buffer
                sock.settimeout(1)  # Short timeout for quick failure
                sock.connect((server_info['host'], server_info['port']))
                self.server_connections[server_id] = sock
                logger.info(f"Successfully established persistent connection to server {server_id}")
            except Exception as e:
                logger.debug(f"Server {server_id} not ready yet: {e}")
    
    def _send_to_server(self, target_server_id: int, data: dict) -> dict:
        """Send data to specified server"""
        if target_server_id not in self.server_connections:
            # Try to reconnect
            try:
                server_info = self.server_config[target_server_id]
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(30)
                sock.connect((server_info['host'], server_info['port']))
                self.server_connections[target_server_id] = sock
            except Exception as e:
                logger.error(f"Unable to connect to server {target_server_id}: {e}")
                return None

        try:
            sock = self.server_connections[target_server_id]
            # Send data
            data_bytes = json.dumps(data).encode()
            sock.sendall(len(data_bytes).to_bytes(4, 'big'))
            sock.sendall(data_bytes)

            # Receive response
            length_bytes = sock.recv(4)
            if not length_bytes:
                raise ConnectionError("Connection closed")

            length = int.from_bytes(length_bytes, 'big')
            response_data = b''
            while len(response_data) < length:
                chunk = sock.recv(min(length - len(response_data), 4096))
                if not chunk:
                    raise ConnectionError("Receive data interrupted")
                response_data += chunk

            return json.loads(response_data.decode())
        except Exception as e:
            logger.error(f"Communication with server {target_server_id} failed: {e}")
            # Remove dead connection
            if target_server_id in self.server_connections:
                self.server_connections[target_server_id].close()
                del self.server_connections[target_server_id]
            return None
    
    def _send_binary_exchange_data(self, target_server_id: int, query_id: str, e_shares: np.ndarray, f_shares: np.ndarray) -> bool:
        """Send binary format exchange data"""
        # Establish all connections on first call
        if not self.connections_established:
            logger.info("First data exchange, establishing persistent connections...")
            self._establish_persistent_connections()
            self.connections_established = True

        # Check if connection exists and is valid
        if target_server_id not in self.server_connections:
            logger.warning(f"No persistent connection to server {target_server_id}, trying to establish new connection...")
            # Try to establish connection
            try:
                server_info = self.server_config[target_server_id]
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                # Increase socket buffer size
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)  # 16MB receive buffer
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024 * 1024)  # 16MB send buffer
                sock.settimeout(30)
                sock.connect((server_info['host'], server_info['port']))
                self.server_connections[target_server_id] = sock
                logger.info(f"Successfully established new connection to server {target_server_id}")
            except Exception as e:
                logger.error(f"Unable to connect to server {target_server_id}: {e}")
                return False
        else:
            logger.debug(f"Using existing persistent connection to server {target_server_id}")
        
        try:
            sock = self.server_connections[target_server_id]

            # Prepare metadata
            metadata = {
                'command': 'binary_exchange_data',
                'query_id': query_id,
                'from_server': self.server_id,
                'e_shares_shape': e_shares.shape,
                'f_shares_shape': f_shares.shape
            }

            # Send metadata
            metadata_bytes = json.dumps(metadata).encode()
            sock.sendall(len(metadata_bytes).to_bytes(4, 'big'))
            sock.sendall(metadata_bytes)

            # Send binary data
            e_bytes = e_shares.tobytes()
            f_bytes = f_shares.tobytes()

            # Send e_shares
            sock.sendall(len(e_bytes).to_bytes(4, 'big'))
            sock.sendall(e_bytes)

            # Send f_shares
            sock.sendall(len(f_bytes).to_bytes(4, 'big'))
            sock.sendall(f_bytes)

            # Receive confirmation
            response_len = int.from_bytes(sock.recv(4), 'big')
            response_data = sock.recv(response_len)
            response = json.loads(response_data.decode())

            return response.get('status') == 'success'
            
        except Exception as e:
            logger.error(f"Failed to send binary data to server {target_server_id}: {e}")
            # Close dead connection
            if target_server_id in self.server_connections:
                try:
                    self.server_connections[target_server_id].close()
                except:
                    pass
                del self.server_connections[target_server_id]

            # Try to reconnect and retry once
            logger.info(f"Trying to re-establish connection to server {target_server_id}...")
            try:
                server_info = self.server_config[target_server_id]
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                # Increase socket buffer size
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)  # 16MB receive buffer
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024 * 1024)  # 16MB send buffer
                sock.settimeout(30)
                sock.connect((server_info['host'], server_info['port']))
                self.server_connections[target_server_id] = sock
                logger.info(f"Connection re-established, retrying to send data...")

                # Re-prepare and send data
                metadata = {
                    'command': 'binary_exchange_data',
                    'query_id': query_id,
                    'from_server': self.server_id,
                    'e_shares_shape': e_shares.shape,
                    'f_shares_shape': f_shares.shape
                }

                metadata_bytes = json.dumps(metadata).encode()
                sock.sendall(len(metadata_bytes).to_bytes(4, 'big'))
                sock.sendall(metadata_bytes)

                e_bytes = e_shares.tobytes()
                f_bytes = f_shares.tobytes()

                sock.sendall(len(e_bytes).to_bytes(4, 'big'))
                sock.sendall(e_bytes)

                sock.sendall(len(f_bytes).to_bytes(4, 'big'))
                sock.sendall(f_bytes)

                response_len = int.from_bytes(sock.recv(4), 'big')
                response_data = sock.recv(response_len)
                response = json.loads(response_data.decode())

                return response.get('status') == 'success'
            except Exception as retry_error:
                logger.error(f"Connection re-establishment and retry failed: {retry_error}")
                return False
    
    def _handle_data_exchange(self, request: Dict) -> Dict:
        """Handle data exchange request"""
        query_id = request.get('query_id')
        from_server = request.get('from_server')

        # Store received data
        e_shares_list = request.get('e_shares')
        f_shares_list = request.get('f_shares')

        if not all([query_id, from_server, e_shares_list is not None, f_shares_list is not None]):
            return {'status': 'error', 'message': 'Missing required parameters'}

        # Convert data format
        e_shares = np.array(e_shares_list, dtype=np.uint64)
        f_shares = np.array(f_shares_list, dtype=np.uint64)

        # Store data
        if query_id not in self.exchange_storage:
            self.exchange_storage[query_id] = {}

        self.exchange_storage[query_id][f'e_shares_{from_server}'] = e_shares
        self.exchange_storage[query_id][f'f_shares_{from_server}'] = f_shares

        logger.info(f"Received data exchange from server {from_server} (query: {query_id})")

        return {'status': 'success'}

    def _exchange_data_with_servers(self, query_id: str, e_shares: np.ndarray, f_shares: np.ndarray) -> tuple:
        """Exchange data with other servers"""
        send_start = time.time()
        
        # Send data to other servers in parallel
        def send_to_server_async(server_id):
            if server_id == self.server_id:
                return None
            start = time.time()
            success = self._send_binary_exchange_data(server_id, query_id, e_shares, f_shares)
            elapsed = time.time() - start
            if success:
                logger.info(f"Successfully sent binary data to server {server_id} (elapsed: {elapsed:.3f}s)")
            else:
                logger.error(f"Failed to send binary data to server {server_id}")
            return server_id, success

        # Use thread pool for parallel sending
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(send_to_server_async, sid) for sid in [1, 2, 3] if sid != self.server_id]
            concurrent.futures.wait(futures)

        send_time = time.time() - send_start
        logger.info(f"Data sending elapsed: {send_time:.3f}s")
        
        # Wait to receive data from other servers
        max_wait = 30  # Wait at most 30 seconds
        start_time = time.time()

        while time.time() - start_time < max_wait:
            # Check if data from all other servers has been received
            received_all = True
            for server_id in [1, 2, 3]:
                if server_id == self.server_id:
                    continue

                if query_id not in self.exchange_storage or \
                   f'e_shares_{server_id}' not in self.exchange_storage[query_id]:
                    received_all = False
                    break

            if received_all:
                # Organize data
                all_e_from_others = {}
                all_f_from_others = {}

                for server_id in [1, 2, 3]:
                    if server_id == self.server_id:
                        continue

                    all_e_from_others[server_id] = self.exchange_storage[query_id][f'e_shares_{server_id}']
                    all_f_from_others[server_id] = self.exchange_storage[query_id][f'f_shares_{server_id}']

                return all_e_from_others, all_f_from_others

            time.sleep(0.1)

        logger.error(f"Timeout waiting for data from other servers (query: {query_id})")
        return {}, {}
    
    def _multiprocess_vdpf_evaluation(self, serialized_key, num_nodes, num_batches):
        """Multi-process VDPF evaluation"""
        # Create shared memory
        shm = shared_memory.SharedMemory(create=True, size=self.node_shares.nbytes)
        shared_array = np.ndarray(self.node_shares.shape, dtype=self.node_shares.dtype, buffer=shm.buf)
        shared_array[:] = self.node_shares[:]

        # Load balancing allocation
        nodes_per_process = num_nodes // self.vdpf_processes
        remaining_nodes = num_nodes % self.vdpf_processes

        # Prepare process arguments
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
                self.node_shares.shape,
                self.node_shares.dtype,
                self.dataset
            )
            process_args.append(args)
            current_node_start = node_end

        # Execute using process pool
        results = self.process_pool.map(evaluate_batch_range_process, process_args)

        # Merge results
        all_selector_shares = {}
        all_vector_shares = {}

        for process_result in results:
            all_selector_shares.update(process_result['selector_shares'])
            all_vector_shares.update(process_result['vector_shares'])

        # Clean up shared memory
        shm.close()
        shm.unlink()

        return all_selector_shares, all_vector_shares
    
    def _save_exchange_data(self, query_id: str, e_shares: np.ndarray, f_shares: np.ndarray):
        """Save exchange data"""
        filename = f"server_{self.server_id}_query_{query_id}_data.npz"
        filepath = os.path.join(self.exchange_dir, filename)

        data_hash = hashlib.md5((e_shares.tobytes() + f_shares.tobytes())).hexdigest()

        np.savez(filepath,
                 e_shares=e_shares,
                 f_shares=f_shares,
                 hash=data_hash)

    def _load_other_servers_data(self, query_id: str, num_nodes: int) -> Tuple[Dict, Dict]:
        """Load data from other servers - enhanced error handling"""
        all_e_from_others = {}
        all_f_from_others = {}

        for other_id in [1, 2, 3]:
            if other_id == self.server_id:
                continue

            filename = f"server_{other_id}_query_{query_id}_data.npz"
            filepath = os.path.join(self.exchange_dir, filename)

            # Increase wait time and retry mechanism
            max_wait = 60  # Increased to 60 seconds
            retry_interval = 0.5

            for i in range(int(max_wait / retry_interval)):
                if os.path.exists(filepath):
                    try:
                        data = np.load(filepath)
                        all_e_from_others[other_id] = data['e_shares']
                        all_f_from_others[other_id] = data['f_shares']
                        logger.info(f"Successfully loaded data from Server {other_id}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load data from Server {other_id}: {e}")
                        if i < int(max_wait / retry_interval) - 1:
                            time.sleep(retry_interval)
                        continue
                time.sleep(retry_interval)
            else:
                logger.warning(f"Timeout waiting for data from Server {other_id}")

        return all_e_from_others, all_f_from_others

    def _file_sync_barrier(self, query_id: str, phase: str):
        """File system synchronization barrier - enhanced error handling"""
        marker_file = f"server_{self.server_id}_query_{query_id}_{phase}_ready"
        marker_path = os.path.join(self.exchange_dir, marker_file)

        with open(marker_path, 'w') as f:
            f.write(str(time.time()))

        # Wait for other servers
        for other_id in [1, 2, 3]:
            if other_id == self.server_id:
                continue

            other_marker = f"server_{other_id}_query_{query_id}_{phase}_ready"
            other_path = os.path.join(self.exchange_dir, other_marker)

            max_wait = 120  # Increased to 120 seconds
            for i in range(max_wait * 10):  # 100ms intervals
                if os.path.exists(other_path):
                    break
                time.sleep(0.1)

            if not os.path.exists(other_path):
                logger.warning(f"Server {other_id} did not complete {phase}")
    
    def _handle_vector_node_query(self, request: Dict) -> Dict:
        """Handle vector-level node query"""
        try:
            serialized_key = request['dpf_key']
            query_id = request.get('query_id', 'unknown')

            logger.info(f"Handling vector-level node query, query ID: {query_id}")

            # Deserialize key
            key = self.dpf_wrapper._deserialize_key(serialized_key)

            # Initialize
            start_time = time.time()
            num_nodes = len(self.node_shares)
            vector_dim = self.node_shares.shape[1]
            result_accumulator = np.zeros(vector_dim, dtype=np.int64)

            # Phase 1: Multi-process VDPF evaluation
            logger.info(f"Phase 1: Multi-process VDPF evaluation ({self.vdpf_processes} processes)...")
            phase1_start = time.time()

            num_batches = (num_nodes + self.cache_batch_size - 1) // self.cache_batch_size

            all_selector_shares, all_vector_shares = self._multiprocess_vdpf_evaluation(
                serialized_key, num_nodes, num_batches)

            phase1_time = time.time() - phase1_start
            logger.info(f"Phase 1 complete, elapsed {phase1_time:.2f}s")

            # File synchronization barrier - temporarily disabled in distributed environment
            # self._file_sync_barrier(query_id, "phase1")
            logger.info("Skipping file synchronization barrier (distributed environment)")
            
            # Phase 2: e/f calculation
            logger.info("Phase 2: e/f calculation...")
            phase2_start = time.time()
            
            all_e_shares = np.zeros(num_nodes, dtype=np.uint64)
            all_f_shares = np.zeros((num_nodes, vector_dim), dtype=np.uint64)
            all_computation_states = {}
            
            for batch_idx in range(num_batches):
                batch_start = batch_idx * self.cache_batch_size
                batch_end = min(batch_start + self.cache_batch_size, num_nodes)
                batch_size = batch_end - batch_start
                
                # Batch get triples
                batch_triples = []
                for _ in range(batch_size):
                    batch_triples.append(self.mult_server.get_next_triple())

                # Process batch
                for local_idx in range(batch_size):
                    global_idx = batch_start + local_idx
                    computation_id = f"query_{query_id}_pos{global_idx}"
                    
                    a, b, c = batch_triples[local_idx]
                    
                    e_share = (all_selector_shares[global_idx] - a) % self.field_size
                    all_e_shares[global_idx] = e_share
                    
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
            logger.info(f"Phase 2 complete, elapsed {phase2_time:.2f}s")

            # Phase 3: Data exchange
            logger.info("Phase 3: Data exchange...")
            phase3_start = time.time()

            # Exchange data with other servers via network
            all_e_from_others, all_f_from_others = self._exchange_data_with_servers(query_id, all_e_shares, all_f_shares)

            phase3_time = time.time() - phase3_start
            logger.info(f"Phase 3 complete, elapsed {phase3_time:.2f}s")

            # Phase 4: Reconstruction computation
            logger.info("Phase 4: Reconstruction computation...")
            phase4_start = time.time()
            
            # Lagrange coefficients
            lagrange_1 = 2
            lagrange_2 = self.field_size - 1

            # Batch reconstruction
            for batch_idx in range(num_batches):
                batch_start = batch_idx * self.cache_batch_size
                batch_end = min(batch_start + self.cache_batch_size, num_nodes)
                batch_size = batch_end - batch_start
                
                # Extract batch data
                batch_e_shares_local = all_e_shares[batch_start:batch_end]
                batch_f_shares_local = all_f_shares[batch_start:batch_end, :]

                # Build shares matrix
                e_shares_matrix = np.zeros((batch_size, 3), dtype=np.uint64)
                e_shares_matrix[:, self.server_id - 1] = batch_e_shares_local

                f_shares_matrix = np.zeros((batch_size, vector_dim, 3), dtype=np.uint64)
                f_shares_matrix[:, :, self.server_id - 1] = batch_f_shares_local

                # Fill in data from other servers
                for other_id, other_e_shares in all_e_from_others.items():
                    e_shares_matrix[:, other_id - 1] = other_e_shares[batch_start:batch_end]

                for other_id, other_f_shares in all_f_from_others.items():
                    f_shares_matrix[:, :, other_id - 1] = other_f_shares[batch_start:batch_end, :]

                # Reconstruct
                batch_e_reconstructed = (e_shares_matrix[:, 0] * lagrange_1 +
                                       e_shares_matrix[:, 1] * lagrange_2) % self.field_size

                batch_f_reconstructed = (f_shares_matrix[:, :, 0] * lagrange_1 +
                                       f_shares_matrix[:, :, 1] * lagrange_2) % self.field_size

                # Get triples
                batch_triples = []
                for local_idx in range(batch_size):
                    global_idx = batch_start + local_idx
                    state = all_computation_states[global_idx]
                    a, b, c = state['triple']
                    batch_triples.append((a, b, c))

                batch_a = np.array([t[0] for t in batch_triples], dtype=np.uint64)
                batch_b = np.array([t[1] for t in batch_triples], dtype=np.uint64)
                batch_c = np.array([t[2] for t in batch_triples], dtype=np.uint64)

                # Calculate result
                batch_e_expanded = batch_e_reconstructed[:, np.newaxis]
                batch_a_expanded = batch_a[:, np.newaxis]
                batch_b_expanded = batch_b[:, np.newaxis]
                batch_c_expanded = batch_c[:, np.newaxis]

                batch_result = batch_c_expanded
                batch_result = (batch_result + batch_e_expanded * batch_b_expanded) % self.field_size
                batch_result = (batch_result + batch_f_reconstructed * batch_a_expanded) % self.field_size
                batch_result = (batch_result + batch_e_expanded * batch_f_reconstructed) % self.field_size

                batch_contribution = np.sum(batch_result, axis=0) % self.field_size
                result_accumulator = (result_accumulator + batch_contribution) % self.field_size

                # Clean up cache
                for local_idx in range(batch_size):
                    global_idx = batch_start + local_idx
                    state = all_computation_states[global_idx]
                    computation_id = state['computation_id']
                    if computation_id in self.mult_server.computation_cache:
                        del self.mult_server.computation_cache[computation_id]

            phase4_time = time.time() - phase4_start
            total_time = time.time() - start_time

            logger.info(f"Query complete:")
            logger.info(f"  Phase 1 (VDPF): {phase1_time:.2f}s")
            logger.info(f"  Phase 2 (e/f): {phase2_time:.2f}s")
            logger.info(f"  Phase 3 (Exchange): {phase3_time:.2f}s")
            logger.info(f"  Phase 4 (Reconstruct): {phase4_time:.2f}s")
            logger.info(f"  Total: {total_time:.2f}s")

            # Complete synchronization
            # self._file_sync_barrier(query_id, "phase4_complete")
            logger.info("Skipping file synchronization barrier (distributed environment)")

            # Clean up files
            self._cleanup_query_files(query_id)

            # Return result
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
            logger.error(f"Query processing error: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}
    
    def _cleanup_query_files(self, query_id: str):
        """Clean up query-related files and exchange data"""
        # Clean up file system (if any)
        for filename in os.listdir(self.exchange_dir):
            if query_id in filename:
                try:
                    os.remove(os.path.join(self.exchange_dir, filename))
                except:
                    pass

        # Clean up exchange data in memory
        if query_id in self.exchange_storage:
            del self.exchange_storage[query_id]
            logger.info(f"Cleaned up exchange data for query {query_id}")

    def _get_status(self) -> Dict:
        """Get server status"""
        return {
            'status': 'success',
            'server_id': self.server_id,
            'mode': 'distributed',
            'host': self.host,
            'port': self.port,
            'dataset': self.dataset,
            'vdpf_processes': self.vdpf_processes,
            'data_loaded': {
                'nodes': self.node_shares.shape
            },
            'triples_available': self.mult_server.triple_array.shape[0] - self.mult_server.used_count if self.mult_server.triple_array is not None else 0
        }


def main():
    """Main function"""
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description='Distributed vector query server')
    parser.add_argument('--server-id', type=int, required=True, choices=[1, 2, 3],
                        help='Server ID (1, 2, or 3)')
    parser.add_argument('--dataset', type=str, default='siftsmall',
                        choices=['laion', 'siftsmall', 'tripclick', 'ms_marco', 'nfcorpus'],
                        help='Dataset name (default: siftsmall)')
    parser.add_argument('--vdpf-processes', type=int, default=4,
                        help='Number of VDPF evaluation processes (default: 4)')

    args = parser.parse_args()

    server = DistributedServer(args.server_id, args.dataset, args.vdpf_processes)
    server.start()


if __name__ == "__main__":
    main()