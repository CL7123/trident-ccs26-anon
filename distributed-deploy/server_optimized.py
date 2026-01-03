#!/usr/bin/env python3
"""
Optimized Distributed Server - Phase 2 and Phase 4 use multiprocessing parallelization
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

# Add project paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('~/trident/src')
sys.path.append('~/trident/standardDPF')
sys.path.append('~/trident/query-opti')

from dpf_wrapper import VDPFVectorWrapper
from dpf_wrapper_optimized import OptimizedVDPFVectorWrapper
from binary_protocol import BinaryProtocol
from basic_functionalities import get_config, MPC23SSS, Share
from secure_multiplication import NumpyMultiplicationServer

# CPU affinity settings
try:
    sys.path.append('~/trident/query-opti')
    from cpu_affinity_optimizer import set_process_affinity
    total_cores = cpu_count()

    # Each server has its own independent 32 physical cores
    CPU_AFFINITY_AVAILABLE = True
except Exception as e:
    print(f"CPU affinity setting is not available: {e}")
    CPU_AFFINITY_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==================== Phase 2 parallel computation function ====================

def phase2_compute_batch(args):
    """Phase 2 batch processing computation function"""
    batch_start, batch_end, selector_shares, data_shares, mult_triples, field_size = args
    
    batch_size = batch_end - batch_start
    vector_dim = data_shares.shape[1]

    # Local computation
    all_e_shares = np.zeros(batch_size, dtype=np.uint64)
    all_f_shares = np.zeros((batch_size, vector_dim), dtype=np.uint64)
    computation_states = {}

    for local_idx in range(batch_size):
        global_idx = batch_start + local_idx

        # Get multiplication triple
        a, b, c = mult_triples[global_idx]

        # Calculate e_share and f_shares
        x_value = int(selector_shares[global_idx])
        y_values = data_shares[global_idx].astype(np.uint64)

        e_value = (x_value - a) % field_size
        all_e_shares[local_idx] = e_value

        f_values = (y_values - b) % field_size
        all_f_shares[local_idx] = f_values

        # Save computation state
        computation_states[global_idx] = {
            'computation_id': f'query_{global_idx}',
            'triple': (a, b, c),
            'a': a,
            'b': b,
            'c': c
        }

    return batch_start, batch_end, all_e_shares, all_f_shares, computation_states


# ==================== Phase 4 parallel computation function ====================

def phase4_reconstruct_batch(args):
    """Phase 4 batch processing reconstruction function"""
    (batch_start, batch_end, e_shares_local, f_shares_local, 
     e_shares_others, f_shares_others, computation_states,
     lagrange_1, lagrange_2, field_size) = args
    
    batch_size = batch_end - batch_start
    vector_dim = f_shares_local.shape[1]

    # Prepare data matrix
    e_shares_matrix = np.zeros((batch_size, 3), dtype=np.uint64)
    e_shares_matrix[:, 0] = e_shares_local

    f_shares_matrix = np.zeros((batch_size, vector_dim, 3), dtype=np.uint64)
    f_shares_matrix[:, :, 0] = f_shares_local

    # Fill data from other servers
    for other_id, other_e in e_shares_others.items():
        e_shares_matrix[:, other_id - 1] = other_e[batch_start:batch_end]

    for other_id, other_f in f_shares_others.items():
        f_shares_matrix[:, :, other_id - 1] = other_f[batch_start:batch_end, :]

    # Lagrange interpolation reconstruction
    e_reconstructed = (e_shares_matrix[:, 0] * lagrange_1 +
                      e_shares_matrix[:, 1] * lagrange_2) % field_size

    f_reconstructed = (f_shares_matrix[:, :, 0] * lagrange_1 +
                      f_shares_matrix[:, :, 1] * lagrange_2) % field_size

    # Get multiplication triples
    batch_a = np.zeros(batch_size, dtype=np.uint64)
    batch_b = np.zeros(batch_size, dtype=np.uint64)
    batch_c = np.zeros(batch_size, dtype=np.uint64)

    for local_idx in range(batch_size):
        global_idx = batch_start + local_idx
        state = computation_states[global_idx]
        batch_a[local_idx] = state['a']
        batch_b[local_idx] = state['b']
        batch_c[local_idx] = state['c']

    # Calculate final result
    e_expanded = e_reconstructed[:, np.newaxis]
    a_expanded = batch_a[:, np.newaxis]
    b_expanded = batch_b[:, np.newaxis]
    c_expanded = batch_c[:, np.newaxis]

    result = c_expanded
    result = (result + e_expanded * b_expanded) % field_size
    result = (result + f_reconstructed * a_expanded) % field_size
    result = (result + e_expanded * f_reconstructed) % field_size

    # Accumulate contributions
    contribution = np.sum(result, axis=0) % field_size

    return contribution


class OptimizedDistributedServer:
    """Optimized distributed server"""
    
    def __init__(self, server_id: int, dataset: str = "siftsmall", vdpf_processes: int = 32):
        self.server_id = server_id
        self.dataset = dataset
        self.config = get_config(dataset)
        self.field_size = self.config.prime
        self.is_query_processing = False

        # Multiprocessing optimization parameters
        self.vdpf_processes = vdpf_processes
        self.phase2_processes = 28  # Phase 2 uses 28 processes
        self.phase4_processes = 28  # Phase 4 uses 28 processes
        self.cache_batch_size = max(100, 1000 // max(vdpf_processes // 4, 1))

        logger.info(f"Using configuration: VDPF processes={vdpf_processes}, Phase 2 processes={self.phase2_processes}, Phase 4 processes={self.phase4_processes}")

        # Create process pools
        self.vdpf_pool = Pool(processes=self.vdpf_processes)
        self.phase2_pool = Pool(processes=self.phase2_processes)
        self.phase4_pool = Pool(processes=self.phase4_processes)

        # Thread pool for I/O operations
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=8,
            thread_name_prefix=f"Server{server_id}-IO"
        )

        # Initialize components
        self.dpf_wrapper = OptimizedVDPFVectorWrapper(dataset_name=dataset)
        self.mpc = MPC23SSS(self.config)
        self.mult_server = NumpyMultiplicationServer(server_id, self.config)

        # Load data
        self._load_data()

        # Network configuration
        self.host = '0.0.0.0'
        self.port = 8000 + server_id
        self.servers_config = self._load_servers_config()

        # Persistent connections
        self.server_connections = {}
        self.connections_established = False
        self.connection_lock = threading.Lock()

        # Data exchange storage
        self.exchange_data = {}
        
    def _load_data(self):
        """Load vector-level secret shared data"""
        logger.info(f"Loading {self.dataset} data...")

        self.data_dir = f"~/trident/dataset/{self.dataset}/server_{self.server_id}"

        # Load node vector shares
        self.nodes_path = os.path.join(self.data_dir, "nodes_shares.npy")
        self.node_shares = np.load(self.nodes_path)
        logger.info(f"Node data: {self.node_shares.shape}")
        logger.info(f"Data size: {self.node_shares.nbytes / 1024 / 1024:.1f}MB")

        # Preload multiplication triples
        logger.info(f"Multiplication triples automatically loaded from local directory")
        logger.info(f"Available triples count: {len(self.mult_server.triple_array) if self.mult_server.triple_array is not None else 0}")
        
    def _load_servers_config(self) -> Dict:
        """Load server configuration"""
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
        """Process query request - optimized version"""
        try:
            if self.is_query_processing:
                return {'status': 'error', 'message': 'Server is busy processing another query'}

            self.is_query_processing = True
            start_time = time.time()
            query_id = query_id or f'query_{int(time.time()*1000)}'

            logger.info(f"Starting to process query {query_id}")

            # Parameters
            num_nodes = len(self.node_shares)
            vector_dim = self.node_shares.shape[1]
            result_accumulator = np.zeros(vector_dim, dtype=np.uint64)

            # Calculate number of batches
            num_batches = (num_nodes + self.cache_batch_size - 1) // self.cache_batch_size

            # ========== Phase 1: VDPF evaluation (multiprocessing) ==========
            logger.info(f"Phase 1: Multiprocessing VDPF evaluation ({self.vdpf_processes} processes)...")
            phase1_start = time.time()

            # Load balancing allocation
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

            # Set environment variable to pass server_id
            os.environ['SERVER_ID'] = str(self.server_id)

            # Execute VDPF evaluation
            all_results = self.vdpf_pool.map(vdpf_evaluate_range_optimized, process_args)

            # Merge results
            selector_values = np.zeros(num_nodes, dtype=np.uint32)
            for node_start, node_end, results in all_results:
                selector_values[node_start:node_end] = results

            phase1_time = time.time() - phase1_start
            logger.info(f"Phase 1 completed, elapsed time {phase1_time:.2f}s")

            # ========== Phase 2: e/f calculation (multiprocessing) ==========
            logger.info(f"Phase 2: Multiprocessing e/f calculation ({self.phase2_processes} processes)...")
            phase2_start = time.time()

            # Prepare batch parameters
            phase2_args = []
            mult_triples = []

            # Pre-fetch all multiplication triples
            for i in range(num_nodes):
                mult_triples.append(self.mult_server.get_next_triple())

            # Calculate batches for each process
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

            # Execute Phase 2 in parallel
            phase2_results = self.phase2_pool.map(phase2_compute_batch, phase2_args)

            # Merge results
            all_e_shares = np.zeros(num_nodes, dtype=np.uint64)
            all_f_shares = np.zeros((num_nodes, vector_dim), dtype=np.uint64)
            all_computation_states = {}

            for batch_start, batch_end, e_shares, f_shares, states in phase2_results:
                all_e_shares[batch_start:batch_end] = e_shares
                all_f_shares[batch_start:batch_end] = f_shares
                all_computation_states.update(states)

            phase2_time = time.time() - phase2_start
            logger.info(f"Phase 2 completed, elapsed time {phase2_time:.2f}s")

            # ========== Phase 3: Data exchange (original implementation) ==========
            logger.info("Phase 3: Data exchange...")
            phase3_start = time.time()

            all_e_from_others, all_f_from_others = self._exchange_data_with_servers(
                query_id, all_e_shares, all_f_shares
            )

            phase3_time = time.time() - phase3_start
            logger.info(f"Phase 3 completed, elapsed time {phase3_time:.2f}s")

            # ========== Phase 4: Reconstruction computation (multiprocessing) ==========
            logger.info(f"Phase 4: Multiprocessing reconstruction computation ({self.phase4_processes} processes)...")
            phase4_start = time.time()

            # Lagrange coefficients
            lagrange_1 = 2
            lagrange_2 = self.field_size - 1

            # Prepare batch parameters
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

            # Execute Phase 4 in parallel
            phase4_results = self.phase4_pool.map(phase4_reconstruct_batch, phase4_args)

            # Accumulate all contributions
            for contribution in phase4_results:
                result_accumulator = (result_accumulator + contribution) % self.field_size

            phase4_time = time.time() - phase4_start
            total_time = time.time() - start_time

            logger.info(f"Query completed:")
            logger.info(f"  Phase 1 (VDPF): {phase1_time:.2f}s")
            logger.info(f"  Phase 2 (e/f): {phase2_time:.2f}s")
            logger.info(f"  Phase 3 (exchange): {phase3_time:.2f}s")
            logger.info(f"  Phase 4 (reconstruction): {phase4_time:.2f}s")
            logger.info(f"  Total: {total_time:.2f}s")

            # Cleanup
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
        finally:
            self.is_query_processing = False
    
    def _exchange_data_with_servers(self, query_id: str, e_shares: np.ndarray, f_shares: np.ndarray):
        """Exchange data with other servers (maintain original implementation)"""
        all_e_from_others = {}
        all_f_from_others = {}

        # Use thread pool to send in parallel
        def send_to_server_async(target_id):
            success = self._send_binary_exchange_data(target_id, query_id, e_shares, f_shares)
            return target_id, success

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(send_to_server_async, sid) for sid in [1, 2, 3] if sid != self.server_id]
            concurrent.futures.wait(futures)

            for future in futures:
                target_id, success = future.result()
                if success:
                    logger.info(f"Successfully sent data to server {target_id}")
                else:
                    logger.error(f"Failed to send data to server {target_id}")

        # Receive data from other servers
        all_e_from_others, all_f_from_others = self._receive_all_exchange_data(query_id)

        return all_e_from_others, all_f_from_others
    
    def _send_binary_exchange_data(self, target_server_id: int, query_id: str,
                                   e_shares: np.ndarray, f_shares: np.ndarray) -> bool:
        """Send binary format exchange data"""
        # Establish all connections on first call
        if not self.connections_established:
            logger.info("First data exchange, establishing persistent connections...")
            self._establish_persistent_connections()
            self.connections_established = True

        # Check connection
        if target_server_id not in self.server_connections:
            logger.warning(f"No persistent connection to server {target_server_id}")
            return False

        conn = self.server_connections[target_server_id]

        try:
            # Send command
            command = {
                'command': 'exchange_data',
                'query_id': query_id,
                'from_server': self.server_id
            }
            command_data = json.dumps(command).encode()
            conn.sendall(len(command_data).to_bytes(4, 'big'))
            conn.sendall(command_data)

            # Prepare binary data
            vector_dim = f_shares.shape[1]
            header = struct.pack('!III', len(e_shares), vector_dim, self.server_id)
            e_bytes = e_shares.astype(np.float32).tobytes()
            f_bytes = f_shares.astype(np.float32).tobytes()

            data = header + e_bytes + f_bytes

            # Send data
            conn.sendall(len(data).to_bytes(4, 'big'))
            conn.sendall(data)

            # Receive acknowledgement
            ack_length_bytes = conn.recv(4)
            if ack_length_bytes:
                ack_length = int.from_bytes(ack_length_bytes, 'big')
                ack_data = conn.recv(ack_length)
                ack = json.loads(ack_data.decode())
                return ack.get('status') == 'received'

            return False

        except Exception as e:
            logger.error(f"Error sending data to server {target_server_id}: {e}")
            return False
    
    def _receive_all_exchange_data(self, query_id: str):
        """Receive exchange data from all other servers"""
        all_e_from_others = {}
        all_f_from_others = {}

        # Wait to receive data
        max_wait_time = 30
        start_wait = time.time()
        expected_servers = [sid for sid in [1, 2, 3] if sid != self.server_id]

        while len(all_e_from_others) < len(expected_servers):
            if time.time() - start_wait > max_wait_time:
                logger.error(f"Timeout waiting for exchange data")
                break

            with self.connection_lock:
                for from_server in expected_servers:
                    if from_server not in all_e_from_others:
                        key = f"{query_id}_from_{from_server}"
                        if key in self.exchange_data:
                            data = self.exchange_data[key]
                            all_e_from_others[from_server] = data['e_shares']
                            all_f_from_others[from_server] = data['f_shares']
                            logger.info(f"Received exchange data from server {from_server}")
                            del self.exchange_data[key]

            if len(all_e_from_others) < len(expected_servers):
                time.sleep(0.01)

        logger.info(f"Received data from {len(all_e_from_others)}/{len(expected_servers)} servers")

        return all_e_from_others, all_f_from_others
    
    def _establish_persistent_connections(self):
        """Establish persistent connections to other servers"""
        for server_id, server_info in self.servers_config.items():
            if server_id != self.server_id:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                    sock.connect((server_info['host'], server_info['port']))

                    self.server_connections[server_id] = sock
                    logger.info(f"Established persistent connection to server {server_id}")

                except Exception as e:
                    logger.error(f"Unable to connect to server {server_id}: {e}")

    def _cleanup_query_files(self, query_id: str):
        """Clean up query-related files"""
        pass

    def _handle_client(self, client_socket, client_address):
        """Handle client connection"""
        logger.info(f"New client connection: {client_address}")
        
        try:
            while True:
                # Receive request length
                length_bytes = client_socket.recv(4)
                if not length_bytes:
                    break

                length = int.from_bytes(length_bytes, 'big')

                # Receive request data
                data = b''
                while len(data) < length:
                    chunk = client_socket.recv(min(length - len(data), 4096))
                    if not chunk:
                        break
                    data += chunk

                if len(data) < length:
                    break

                # Parse request
                try:
                    # Check if it is binary protocol (by first byte)
                    if len(data) > 0 and data[0] in [BinaryProtocol.CMD_QUERY_NODE_VECTOR, BinaryProtocol.CMD_GET_STATUS]:
                        request = BinaryProtocol.decode_request(data)
                        logger.info(f"Received binary request: {request.get('command', 'unknown')}")

                        if request.get('command') == 'query_node_vector':
                            dpf_key = request.get('dpf_key')
                            query_id = request.get('query_id')
                            response = self.process_query(dpf_key, query_id)
                            response_data = BinaryProtocol.encode_response(response)
                            client_socket.sendall(response_data)
                    else:
                        # JSON request
                        request = json.loads(data.decode())
                        logger.info(f"Received JSON request: {request.get('command', 'unknown')}")

                        # Special handling for data exchange request
                        if request.get('command') == 'exchange_data':
                            self._handle_exchange_data(request, client_socket)
                        else:
                            response = self._handle_request(request)
                            response_data = json.dumps(response).encode()
                            client_socket.sendall(len(response_data).to_bytes(4, 'big'))
                            client_socket.sendall(response_data)

                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing failed: {e}")
                    error_response = {'status': 'error', 'message': 'Invalid JSON format'}
                    error_data = json.dumps(error_response).encode()
                    client_socket.sendall(len(error_data).to_bytes(4, 'big'))
                    client_socket.sendall(error_data)
                except Exception as e:
                    logger.error(f"Request parsing failed: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error handling client request: {e}")
        finally:
            client_socket.close()
            logger.info(f"Client connection closed: {client_address}")
    
    def _handle_request(self, request: Dict) -> Dict:
        """Handle non-query requests"""
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
            # Handle data exchange request
            return self._handle_exchange_data(request)

        else:
            return {'status': 'error', 'message': f'Unknown command: {command}'}
    
    def _handle_exchange_data(self, request: Dict, client_socket) -> None:
        """Handle data exchange request"""
        query_id = request.get('query_id')
        from_server = request.get('from_server')

        try:
            # Receive binary data length
            length_bytes = client_socket.recv(4)
            if not length_bytes:
                return

            length = int.from_bytes(length_bytes, 'big')

            # Receive binary data
            data = b''
            while len(data) < length:
                chunk = client_socket.recv(min(length - len(data), 65536))
                if not chunk:
                    break
                data += chunk

            # Parse binary data
            header_size = 12  # 3 * 4 bytes for III format
            num_nodes, vector_dim, server_id = struct.unpack('!III', data[:header_size])

            # Extract data
            offset = header_size
            e_size = num_nodes * 4  # float32
            f_size = num_nodes * vector_dim * 4  # float32

            e_bytes = data[offset:offset + e_size]
            f_bytes = data[offset + e_size:offset + e_size + f_size]

            # Convert to numpy arrays
            e_shares = np.frombuffer(e_bytes, dtype=np.float32).astype(np.uint64)
            f_shares = np.frombuffer(f_bytes, dtype=np.float32).reshape(num_nodes, vector_dim).astype(np.uint64)

            # Store data
            key = f"{query_id}_from_{from_server}"
            with self.connection_lock:
                self.exchange_data[key] = {
                    'e_shares': e_shares,
                    'f_shares': f_shares
                }

            logger.info(f"Received exchange data from server {from_server}: e_shares shape={e_shares.shape}, f_shares shape={f_shares.shape}")

            # Send acknowledgement
            ack = {'status': 'received'}
            ack_data = json.dumps(ack).encode()
            client_socket.sendall(len(ack_data).to_bytes(4, 'big'))
            client_socket.sendall(ack_data)

        except Exception as e:
            logger.error(f"Error handling exchange data: {e}")
            import traceback
            traceback.print_exc()
    
    def start(self):
        """Start server"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)

        logger.info(f"Optimized server {self.server_id} starting")
        logger.info(f"Listening address: {self.host}:{self.port}")
        logger.info(f"Dataset: {self.dataset}")
        logger.info(f"VDPF processes: {self.vdpf_processes}")
        logger.info(f"Phase 2 processes: {self.phase2_processes}")
        logger.info(f"Phase 4 processes: {self.phase4_processes}")

        # Warm up process pools
        logger.info("Warming up process pools...")
        warmup_start = time.time()

        # Warm up all process pools
        self.vdpf_pool.map(warmup_process, range(self.vdpf_processes))
        self.phase2_pool.map(warmup_process, range(self.phase2_processes))
        self.phase4_pool.map(warmup_process, range(self.phase4_processes))

        warmup_time = time.time() - warmup_start
        logger.info(f"Process pool warmup completed, elapsed time {warmup_time:.2f}s")

        logger.info("Server ready, waiting for connections...")

        try:
            while True:
                client_socket, client_address = server_socket.accept()
                # Use thread pool to handle clients
                self.executor.submit(self._handle_client, client_socket, client_address)

        except KeyboardInterrupt:
            logger.info("Server shutting down...")
        finally:
            server_socket.close()
            self.vdpf_pool.close()
            self.phase2_pool.close()
            self.phase4_pool.close()
            self.executor.shutdown()


# Copy necessary auxiliary functions from original file
def warmup_process(process_id):
    """Warm up process by loading necessary modules"""
    import time
    import sys
    sys.path.append('~/trident/query-opti')
    from dpf_wrapper_optimized import OptimizedVDPFVectorWrapper

    # Simple computation task for warmup
    _ = sum(i * i for i in range(1000))
    return process_id


def vdpf_evaluate_range_optimized(args):
    """Optimized VDPF evaluation function"""
    dpf_key, process_id, node_start, node_end = args

    # CPU affinity settings (can be added here if needed)
    # Skipped for now due to correct function signature requirement

    from dpf_wrapper_optimized import OptimizedVDPFVectorWrapper
    from binary_serializer import BinaryKeySerializer

    dataset_name = os.environ.get('DATASET_NAME', 'siftsmall')
    wrapper = OptimizedVDPFVectorWrapper(dataset_name=dataset_name)

    # Deserialize key
    if isinstance(dpf_key, bytes):
        key = BinaryKeySerializer.deserialize_vdpf23_key(dpf_key)
    else:
        key = wrapper._deserialize_key(dpf_key)

    # Get actual server_id
    server_id = int(os.environ.get('SERVER_ID', '1'))

    # Use eval_batch method
    batch_results = wrapper.eval_batch(key, node_start, node_end, server_id)

    # Extract results
    results = []
    for node_id in range(node_start, node_end):
        if node_id in batch_results:
            results.append(batch_results[node_id])
        else:
            logger.error(f"Process {process_id}: Missing result for node {node_id}")
            results.append(0)

    return node_start, node_end, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimized distributed server')
    parser.add_argument('--server-id', type=int, required=True, help='Server ID (1, 2, or 3)')
    parser.add_argument('--dataset', type=str, default='siftsmall', help='Dataset name')
    parser.add_argument('--vdpf-processes', type=int, default=32, help='Number of VDPF evaluation processes')

    args = parser.parse_args()

    # Set environment variable
    os.environ['DATASET_NAME'] = args.dataset

    server = OptimizedDistributedServer(args.server_id, args.dataset, args.vdpf_processes)
    server.start()