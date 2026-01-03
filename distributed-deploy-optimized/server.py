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
from config import SERVERS, SERVER_TO_SERVER

# Set logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [Server] %(message)s')
logger = logging.getLogger(__name__)

# Global function for process pool invocation
def warmup_process(process_id):
    """Warm up process, load necessary modules"""
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
        
        # Each server has its own independent 64 cores, use process_id directly as core ID
        if process_id < total_cores:
            import os
            pid = os.getpid()
            os.sched_setaffinity(pid, {process_id})
            print(f"[Server {server_id}, Process {process_id}] Bind to core {process_id}")
        else:
            # If process count exceeds core count, assign in round-robin fashion
            core_id = process_id % total_cores
            import os
            pid = os.getpid()
            os.sched_setaffinity(pid, {core_id})
            print(f"[Server {server_id}, Process {process_id}] Bind to core {core_id}")
    except Exception as e:
        print(f"[Process {process_id}] CPU binding failed: {e}")
    
    process_total_start = time.time()
    
    # Calculate actual number of nodes to process
    actual_nodes = 0
    for batch_idx in range(start_batch, end_batch):
        batch_start = batch_idx * cache_batch_size
        batch_end = min(batch_start + cache_batch_size, num_nodes)
        actual_nodes += (batch_end - batch_start)
    
    # VDPF instance creation
    from dpf_wrapper_optimized import OptimizedVDPFVectorWrapper
    dpf_wrapper = OptimizedVDPFVectorWrapper(dataset_name=dataset_name)
    
    # Key deserialization
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
    """Distributed server in real network environment"""
    
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

        # Clean old files
        self._cleanup_old_files()

        # Multi-process optimization parameters
        # Dynamically adjust batch size based on process count, each process handles fewer batches to improve parallelism
        self.cache_batch_size = max(100, 1000 // max(vdpf_processes // 4, 1))
        self.vdpf_processes = vdpf_processes

        # Concurrent optimization: Limit processes per query to support multiple concurrent queries
        # For example: 64 total processes, 16 per query, can handle 4 concurrent queries
        self.processes_per_query = min(16, vdpf_processes)
        self.max_concurrent_queries = vdpf_processes // self.processes_per_query

        self.worker_threads = 8  # Increase worker thread count
        logger.info(f"Using batch size: {self.cache_batch_size}, VDPF process pool: {self.vdpf_processes}")
        logger.info(f"Processes per query: {self.processes_per_query}, Max concurrent queries: {self.max_concurrent_queries}")
        
        # Create thread pool
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.worker_threads,
            thread_name_prefix=f"Server{server_id}-General"
        )
        
        # Create process pool
        self.process_pool = Pool(processes=self.vdpf_processes)

        # Data exchange storage
        self.exchange_storage = {}  # {query_id: {'e_shares': ..., 'f_shares': ...}}

        # Detailed performance data storage (for profiling)
        self.query_profiling = {}  # {query_id: {'phase3_details': {...}}}

        # Inter-server connection management
        self.server_connections = {}  # {server_id: socket}
        self.server_config = SERVER_TO_SERVER  # Inter-server communication uses private IPs

        # Network transmission statistics
        self.network_stats = {
            'total_bytes_sent': 0,
            'total_bytes_received': 0,
            'transfer_count': 0,
            'transfer_details': []  # Detailed information for each transfer
        }
        
        logger.info(f"Distributed server initialization complete")
        logger.info(f"Listening address: {self.host}:{self.port}")
        logger.info(f"Dataset: {self.dataset}")
        logger.info(f"Number of VDPF processes: {self.vdpf_processes}")
        
        # Warm up process pool
        self._warmup_process_pool()

        # Delay establishing connections
        self.connections_established = False
        
    def _cleanup_old_files(self):
        """Clean old exchange files"""
        try:
            for filename in os.listdir(self.exchange_dir):
                if f"server_{self.server_id}_" in filename:
                    try:
                        os.remove(os.path.join(self.exchange_dir, filename))
                    except:
                        pass
            logger.info("Cleaned old synchronization files")
        except:
            pass
    
    def _warmup_process_pool(self):
        """Warm up process pool"""
        logger.info("Warm up process pool...")
        warmup_start = time.time()
        results = self.process_pool.map(warmup_process, range(self.vdpf_processes))
        warmup_time = time.time() - warmup_start
        logger.info(f"Process pool warmup complete, elapsed time {warmup_time:.2f} seconds")
    
    def _load_data(self):
        """Load vector-level secret share data"""
        logger.info(f"Loading {self.dataset} data...")

        self.data_dir = f"~/trident/dataset/{self.dataset}/server_{self.server_id}"

        # Load node vector shares
        self.nodes_path = os.path.join(self.data_dir, "nodes_shares.npy")
        self.node_shares = np.load(self.nodes_path)
        logger.info(f"Node data: {self.node_shares.shape}")
        logger.info(f"Data size: {self.node_shares.nbytes / 1024 / 1024:.1f}MB")
        logger.info(f"Data type: {self.node_shares.dtype}")
        logger.info(f"Element size: {self.node_shares.itemsize} bytes")
        
        logger.info(f"Triples available: {self.mult_server.triple_array.shape[0] - self.mult_server.used_count if self.mult_server.triple_array is not None else 0}")
    
    def start(self):
        """Start server"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Set socket options to improve network performance
        server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        
        try:
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)
            logger.info(f"Server started successfully, listening {self.host}:{self.port}")
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
        """Handle client request"""
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
                    # Check if it is binary protocol
                    if data[0] in [BinaryProtocol.CMD_QUERY_NODE_VECTOR, BinaryProtocol.CMD_GET_STATUS]:
                        request = BinaryProtocol.decode_request(data)
                        logger.info(f"Received binary request: {request.get('command', 'unknown')}")
                    else:
                        request = json.loads(data.decode())
                        logger.info(f"Received JSON request: {request.get('command', 'unknown')}")
                except Exception as e:
                    logger.error(f"Parse request failed: {e}")
                    continue

                # Special handling for binary data exchange
                if request.get('command') == 'binary_exchange_data':
                    receive_start_time = time.time()
                    from_server = request.get('from_server')
                    query_id = request.get('query_id')

                    logger.info(f"Receiving binary data exchange from server {from_server} (query: {query_id})")
                    logger.info(f"  Expected shape: e_shares={request.get('e_shares_shape')}, f_shares={request.get('f_shares_shape')}")

                    # Receive e_shares - use MSG_WAITALL to eliminate buffer loops
                    e_len_bytes = client_socket.recv(4, socket.MSG_WAITALL)
                    if len(e_len_bytes) != 4:
                        logger.error(f"Receive e_shares length header failed: {len(e_len_bytes)} bytes")
                        continue

                    e_len = int.from_bytes(e_len_bytes, 'big')
                    logger.info(f"  Receiving e_shares: {e_len:,} bytes")

                    e_data = client_socket.recv(e_len, socket.MSG_WAITALL)
                    if len(e_data) != e_len:
                        logger.error(f"e_shares receive incomplete: {len(e_data)}/{e_len} bytes")
                        continue

                    # Receive f_shares - use MSG_WAITALL to eliminate buffer loops
                    f_len_bytes = client_socket.recv(4, socket.MSG_WAITALL)
                    if len(f_len_bytes) != 4:
                        logger.error(f"Receive f_shares length header failed: {len(f_len_bytes)} bytes")
                        continue

                    f_len = int.from_bytes(f_len_bytes, 'big')
                    logger.info(f"  Receiving f_shares: {f_len:,} bytes")

                    f_data = client_socket.recv(f_len, socket.MSG_WAITALL)
                    if len(f_data) != f_len:
                        logger.error(f"f_shares receive incomplete: {len(f_data)}/{f_len} bytes")
                        continue

                    total_received_size = len(data) + len(e_data) + len(f_data) + 8  # +8 for length headers
                    logger.info(f"  Total received: {total_received_size:,} bytes ({total_received_size/1024/1024:.2f} MB)")

                    # Reconstruct array
                    e_shares = np.frombuffer(e_data, dtype=np.uint64)
                    f_shape = tuple(request['f_shares_shape'])
                    logger.debug(f"  Pre-reconstruction check: e_data divisible by 8? {len(e_data) % 8 == 0}, f_data divisible by 8? {len(f_data) % 8 == 0}")
                    logger.debug(f"  Expected uint64 element count: e={len(e_data)//8}, f={len(f_data)//8}")
                    logger.debug(f"  Elements required for f_shape: {np.prod(f_shape)}")
                    
                    f_shares = np.frombuffer(f_data, dtype=np.uint64).reshape(f_shape)
                    
                    # Store data
                    if query_id not in self.exchange_storage:
                        self.exchange_storage[query_id] = {}
                    
                    self.exchange_storage[query_id][f'e_shares_{from_server}'] = e_shares
                    self.exchange_storage[query_id][f'f_shares_{from_server}'] = f_shares

                    # Calculate receive statistics
                    receive_end_time = time.time()
                    receive_duration = receive_end_time - receive_start_time
                    receive_speed = total_received_size / receive_duration if receive_duration > 0 else 0

                    # Store receive details for profiling
                    if query_id not in self.query_profiling:
                        self.query_profiling[query_id] = {
                            'phase3_send': {},
                            'phase3_receive': {},
                            'phase3_receive_start': receive_start_time
                        }

                    self.query_profiling[query_id]['phase3_receive'][from_server] = {
                        'start_time': receive_start_time,
                        'end_time': receive_end_time,
                        'duration_ms': receive_duration * 1000,
                        'data_received_bytes': total_received_size,
                        'speed_mbps': (receive_speed * 8) / (1024 * 1024)
                    }

                    # Update statistics
                    self.network_stats['total_bytes_received'] += total_received_size

                    logger.info(f"Data receive complete:")
                    logger.info(f"  Source: server {from_server}")
                    logger.info(f"  Elapsed time: {receive_duration:.3f} seconds")
                    logger.info(f"  Speed: {receive_speed/1024/1024:.2f} MB/s ({(receive_speed*8)/1024/1024:.2f} Mbps)")
                    
                    # Send confirmation response
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
            logger.info(f"Client {address} disconnected connection")
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
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # Disable Nagle algorithm
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)  # Enable keepalive
                # Increase socket buffer size
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)  # 16MB receive buffer
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024 * 1024)  # 16MB send buffer
                sock.settimeout(1)  # Short timeout, fail fast
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
    
    def _send_binary_exchange_data(self, target_server_id: int, query_id: str, e_shares: np.ndarray, f_shares: np.ndarray) -> dict:
        """
        Send binary format exchange data

        Phase 1 optimization: Use temporary connection instead of shared connection
        - Create independent socket connection for each query
        - Avoid multiple threads competing for the same socket
        - Close connection immediately after send completes
        """
        send_start_time = time.time()
        sock = None

        try:
            # Create temporary connection (independent for each query)
            logger.debug(f"Creating temporary connection to server {target_server_id} for query {query_id}")
            server_info = self.server_config[target_server_id]
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # Increase socket buffer size
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)  # 16MB
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024 * 1024)  # 16MB
            sock.settimeout(30)
            sock.connect((server_info['host'], server_info['port']))
            logger.debug(f"Successfully connected to server {target_server_id}")

            # Prepare metadata
            metadata = {
                'command': 'binary_exchange_data',
                'query_id': query_id,
                'from_server': self.server_id,
                'e_shares_shape': e_shares.shape,
                'f_shares_shape': f_shares.shape
            }

            # Calculate data size
            e_bytes = e_shares.tobytes()
            f_bytes = f_shares.tobytes()
            metadata_bytes = json.dumps(metadata).encode()

            # Send metadata
            sock.sendall(len(metadata_bytes).to_bytes(4, 'big'))
            sock.sendall(metadata_bytes)

            total_data_size = len(metadata_bytes) + len(e_bytes) + len(f_bytes) + 12  # +12 for length headers

            logger.info(f"Sending data to server {target_server_id}:")
            logger.info(f"  e_shares: {e_shares.shape}, {len(e_bytes):,} bytes")
            logger.info(f"  f_shares: {f_shares.shape}, {len(f_bytes):,} bytes")
            logger.info(f"  Metadata: {len(metadata_bytes):,} bytes")
            logger.info(f"  Total size: {total_data_size:,} bytes ({total_data_size/1024/1024:.2f} MB)")

            # Send e_shares
            sock.sendall(len(e_bytes).to_bytes(4, 'big'))
            sock.sendall(e_bytes)

            # Send f_shares
            sock.sendall(len(f_bytes).to_bytes(4, 'big'))
            sock.sendall(f_bytes)

            # Receive confirmation (loop to ensure completeness)
            # Receive length header (4 bytes)
            length_bytes = sock.recv(4)
            if not length_bytes or len(length_bytes) < 4:
                raise ConnectionError("Failed to receive response length")

            response_len = int.from_bytes(length_bytes, 'big')

            # Loop receive complete response data
            response_data = b''
            while len(response_data) < response_len:
                chunk = sock.recv(min(response_len - len(response_data), 4096))
                if not chunk:
                    raise ConnectionError("Response data reception interrupted")
                response_data += chunk

            response = json.loads(response_data.decode())

            # Calculate transmission statistics
            send_duration = time.time() - send_start_time
            transfer_speed = total_data_size / send_duration if send_duration > 0 else 0

            # Update statistics
            self.network_stats['total_bytes_sent'] += total_data_size
            self.network_stats['transfer_count'] += 1
            self.network_stats['transfer_details'].append({
                'target_server': target_server_id,
                'query_id': query_id,
                'data_size_bytes': total_data_size,
                'data_size_mb': total_data_size / 1024 / 1024,
                'duration_seconds': send_duration,
                'speed_mbps': (transfer_speed * 8) / (1024 * 1024),  # Convert to Mbps
                'speed_mb_per_sec': transfer_speed / (1024 * 1024),   # MB/s
                'timestamp': time.time()
            })

            logger.info(f"Data transmission complete:")
            logger.info(f"  Elapsed time: {send_duration:.3f} seconds")
            logger.info(f"  Speed: {transfer_speed/1024/1024:.2f} MB/s ({(transfer_speed*8)/1024/1024:.2f} Mbps)")

            # Return detailed send statistics
            return {
                'success': response.get('status') == 'success',
                'target_server': target_server_id,
                'duration_ms': send_duration * 1000,
                'data_sent_bytes': total_data_size,
                'speed_mbps': (transfer_speed * 8) / (1024 * 1024)
            }

        except Exception as e:
            logger.error(f"Failed to send binary data to server {target_server_id}: {e}")
            return {
                'success': False,
                'target_server': target_server_id,
                'error': str(e)
            }

        finally:
            # Always close temporary connection
            if sock:
                try:
                    sock.close()
                    logger.debug(f"Closed temporary connection to server {target_server_id}")
                except Exception as e:
                    logger.warning(f"Error closing socket: {e}")
    
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

        # Initialize profiling storage for this query
        if query_id not in self.query_profiling:
            self.query_profiling[query_id] = {
                'phase3_send': {},
                'phase3_receive': {},
                'phase3_receive_start': time.time()
            }

        # Send data to other servers in parallel
        def send_to_server_async(server_id):
            if server_id == self.server_id:
                return None
            start = time.time()
            result = self._send_binary_exchange_data(server_id, query_id, e_shares, f_shares)
            elapsed = time.time() - start

            # Store send details for profiling
            self.query_profiling[query_id]['phase3_send'][server_id] = {
                'start_time': start,
                'end_time': time.time(),
                'duration_ms': result.get('duration_ms', elapsed * 1000),
                'data_sent_bytes': result.get('data_sent_bytes', 0),
                'speed_mbps': result.get('speed_mbps', 0),
                'success': result.get('success', False)
            }

            if result.get('success'):
                logger.info(f"Successfully sent binary data to server {server_id} (Elapsed time: {elapsed:.3f} seconds)")
            else:
                logger.error(f"Failed to send binary data to server {server_id}")
            return server_id, result.get('success', False)

        # Use thread pool for parallel sending (increase concurrency)
        # Each query needs to send to 2 servers, supporting 16 concurrent queries requires 32 threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(send_to_server_async, sid) for sid in [1, 2, 3] if sid != self.server_id]
            concurrent.futures.wait(futures)

        send_time = time.time() - send_start
        logger.info(f"Sending data elapsed time: {send_time:.3f} seconds")

        # Wait to receive data from other servers
        max_wait = 30  # Wait up to 30 seconds
        start_time = time.time()

        while time.time() - start_time < max_wait:
            # Check if data from all other servers received
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

        logger.error(f"Timeout waiting for other servers data (query: {query_id})")
        return {}, {}
    
    def _multiprocess_vdpf_evaluation(self, serialized_key, num_nodes, num_batches):
        """Multi-process VDPF evaluation (concurrent optimized version)"""
        # Create shared memory
        shm = shared_memory.SharedMemory(create=True, size=self.node_shares.nbytes)
        shared_array = np.ndarray(self.node_shares.shape, dtype=self.node_shares.dtype, buffer=shm.buf)
        shared_array[:] = self.node_shares[:]

        # Limit processes per query to support true concurrent multi-query
        # For example: 64 processes, 32 per query, can handle 2 concurrent queries
        # Each query slightly slower, but total throughput significantly improved
        processes_to_use = min(32, self.vdpf_processes)

        # Load balancing distribution
        nodes_per_process = num_nodes // processes_to_use
        remaining_nodes = num_nodes % processes_to_use

        # Prepare process arguments
        process_args = []
        current_node_start = 0

        for process_id in range(processes_to_use):
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

        # Execute using process pool (non-blocking submission)
        # map_async allows multiple queries to submit tasks to process pool concurrently
        # Process pool scheduler will automatically assign available processes to tasks
        async_result = self.process_pool.map_async(evaluate_batch_range_process, process_args)

        # Wait for results (during waiting, other queries can submit tasks)
        # Process pool can interleave execution of tasks from multiple queries
        results = async_result.get()

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
        """Load other servers data - Enhanced fault tolerance"""
        all_e_from_others = {}
        all_f_from_others = {}

        for other_id in [1, 2, 3]:
            if other_id == self.server_id:
                continue

            filename = f"server_{other_id}_query_{query_id}_data.npz"
            filepath = os.path.join(self.exchange_dir, filename)

            # Increase wait time and retry mechanism
            max_wait = 60  # Increase to 60 seconds
            retry_interval = 0.5

            for i in range(int(max_wait / retry_interval)):
                if os.path.exists(filepath):
                    try:
                        data = np.load(filepath)
                        all_e_from_others[other_id] = data['e_shares']
                        all_f_from_others[other_id] = data['f_shares']
                        logger.info(f"Successfully loaded Server {other_id} data")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load Server {other_id} data: {e}")
                        if i < int(max_wait / retry_interval) - 1:
                            time.sleep(retry_interval)
                        continue
                time.sleep(retry_interval)
            else:
                logger.warning(f"Timeout waiting for Server {other_id} data")

        return all_e_from_others, all_f_from_others

    def _file_sync_barrier(self, query_id: str, phase: str):
        """File system synchronization barrier - Enhanced fault tolerance"""
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

            max_wait = 120  # Increase to 120 seconds
            for i in range(max_wait * 10):  # 100ms interval
                if os.path.exists(other_path):
                    break
                time.sleep(0.1)

            if not os.path.exists(other_path):
                logger.warning(f"Server {other_id} not completed {phase}")
    
    def _handle_vector_node_query(self, request: Dict) -> Dict:
        """Handle vector-level node query"""
        try:
            serialized_key = request['dpf_key']
            query_id = request.get('query_id', 'unknown')
            
            logger.info(f"Handle vector-level node query，Query ID: {query_id}")
            
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
            logger.info(f"Phase 1 complete, elapsed time {phase1_time:.2f} seconds")

            # File synchronization barrier - Temporarily disabled in distributed environment
            # self._file_sync_barrier(query_id, "phase1")
            logger.info("Skipping file synchronization barrier (distributed environment)")

            # Phase 2: e/f computation
            logger.info("Phase 2: e/f computation...")
            phase2_start = time.time()
            
            all_e_shares = np.zeros(num_nodes, dtype=np.uint64)
            all_f_shares = np.zeros((num_nodes, vector_dim), dtype=np.uint64)
            all_computation_states = {}
            
            logger.info(f"Initialize e_shares: dtype={all_e_shares.dtype}, shape={all_e_shares.shape}")
            logger.info(f"Initialize f_shares: dtype={all_f_shares.dtype}, shape={all_f_shares.shape}")
            
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
            logger.info(f"Phase 2 complete, elapsed time {phase2_time:.2f} seconds")

            # Phase 3: Data exchange
            logger.info("Phase 3: Data exchange...")
            phase3_start = time.time()

            # Exchange data with other servers via network
            all_e_from_others, all_f_from_others = self._exchange_data_with_servers(query_id, all_e_shares, all_f_shares)

            phase3_time = time.time() - phase3_start
            logger.info(f"Phase 3 complete, elapsed time {phase3_time:.2f} seconds")

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
                
                # Build share matrix
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
                
                # Clean cache
                for local_idx in range(batch_size):
                    global_idx = batch_start + local_idx
                    state = all_computation_states[global_idx]
                    computation_id = state['computation_id']
                    if computation_id in self.mult_server.computation_cache:
                        del self.mult_server.computation_cache[computation_id]
            
            phase4_time = time.time() - phase4_start
            total_time = time.time() - start_time

            logger.info(f"Query complete:")
            logger.info(f"  Phase 1 (VDPF): {phase1_time:.2f} seconds")
            logger.info(f"  Phase 2 (e/f): {phase2_time:.2f} seconds")
            logger.info(f"  Phase 3 (Exchange): {phase3_time:.2f} seconds")
            logger.info(f"  Phase 4 (Reconstruct): {phase4_time:.2f} seconds")
            logger.info(f"  Total: {total_time:.2f} seconds")

            # Output network transmission statistics
            network_stats = self._get_network_stats_summary()
            if network_stats:
                logger.info(f"Network transmission statistics:")
                logger.info(f"  Sent data: {network_stats['total_data_sent_mb']:.2f} MB")
                logger.info(f"  Received data: {network_stats['total_data_received_mb']:.2f} MB")
                logger.info(f"  Average transmission speed: {network_stats['avg_speed_mb_per_sec']:.2f} MB/s ({network_stats['avg_speed_mbps']:.2f} Mbps)")
                logger.info(f"  Number of transfers: {network_stats['total_transfers']}")

            # Complete synchronization
            # self._file_sync_barrier(query_id, "phase4_complete")
            logger.info("Skipping file synchronization barrier (distributed environment)")
            
            # Clean files
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
                },
                'detailed_profiling': self.query_profiling.get(query_id, {})
            }

            # Clean up profiling data after query completes
            if query_id in self.query_profiling:
                del self.query_profiling[query_id]

            return response
            
        except Exception as e:
            logger.error(f"Query handling error: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}

    def _cleanup_query_files(self, query_id: str):
        """Clean up query-related files and exchange data"""
        # Clean file system (if any)
        for filename in os.listdir(self.exchange_dir):
            if query_id in filename:
                try:
                    os.remove(os.path.join(self.exchange_dir, filename))
                except:
                    pass

        # Clean exchange data in memory
        if query_id in self.exchange_storage:
            del self.exchange_storage[query_id]
            logger.info(f"Cleaned query {query_id} exchange data")

    def _get_network_stats_summary(self) -> Dict:
        """Get network transmission statistics summary"""
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
            'transfer_details': transfers[-5:]  # Last 5 transfer details
        }
    
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
            'triples_available': self.mult_server.triple_array.shape[0] - self.mult_server.used_count if self.mult_server.triple_array is not None else 0,
            'network_stats': self._get_network_stats_summary()
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