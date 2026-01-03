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


# Global function for process pool calls (must be defined at module level)
def evaluate_batch_range_process(args):
    """Worker function for process pool: evaluate VDPF for specified batch range"""
    process_id, start_batch, end_batch, cache_batch_size, num_nodes, serialized_key, server_id, node_shares_path, dataset_name = args

    print(f"[Process {process_id}] Starting batch evaluation {start_batch}-{end_batch-1}")

    # Each process creates independent VDPF instance
    from dpf_wrapper import VDPFVectorWrapper
    dpf_wrapper = VDPFVectorWrapper(dataset_name=dataset_name)

    # Deserialize key
    key = dpf_wrapper._deserialize_key(serialized_key)

    # Load node data (each process loads independently)
    node_shares = np.load(node_shares_path)

    local_selector_shares = {}
    local_vector_shares = {}
    process_start_time = time.time()

    for batch_idx in range(start_batch, end_batch):
        batch_start = batch_idx * cache_batch_size
        batch_end = min(batch_start + cache_batch_size, num_nodes)
        batch_size = batch_end - batch_start

        # Preload entire batch into cache
        batch_data = node_shares[batch_start:batch_end].copy()

        # Process all nodes in this batch sequentially
        for local_idx in range(batch_size):
            global_idx = batch_start + local_idx

            # VDPF evaluation
            selector_share = dpf_wrapper.eval_at_position(key, global_idx, server_id)
            local_selector_shares[global_idx] = selector_share

            # Use preloaded data
            local_vector_shares[global_idx] = batch_data[local_idx]

    process_time = time.time() - process_start_time
    evaluated_nodes = sum(min(cache_batch_size, num_nodes - b * cache_batch_size)
                        for b in range(start_batch, end_batch))
    ops_per_sec = evaluated_nodes / process_time if process_time > 0 else 0

    print(f"[Process {process_id}] Complete: {process_time:.3f}s, "
          f"{evaluated_nodes} nodes, {ops_per_sec:.0f} ops/sec")

    return {
        'selector_shares': local_selector_shares,
        'vector_shares': local_vector_shares
    }


class MemoryMappedOptimizedServer:
    """Multiprocess data locality optimization server (supports configurable process count)"""

    def __init__(self, server_id: int, dataset: str = "laion", vdpf_processes: int = 4):
        self.server_id = server_id
        self.dataset = dataset
        self.config = get_config(dataset)
        self.field_size = self.config.prime
        self.mpc = MPC23SSS(self.config)

        # Network configuration
        self.host = f"192.168.50.2{server_id}"
        self.port = 8000 + server_id

        # Initialize components
        self.dpf_wrapper = VDPFVectorWrapper(dataset_name=dataset)
        self.mult_server = NumpyMultiplicationServer(server_id, self.config)

        # Load data
        self._load_data()

        # Simulated exchange directory (fallback to standard file system)
        self.exchange_dir = "/tmp/mpc_exchange"
        os.makedirs(self.exchange_dir, exist_ok=True)

        # ===== Multiprocess optimization parameters =====
        self.cache_batch_size = 2000  # Cache-friendly batch size
        self.vdpf_processes = vdpf_processes  # Configurable VDPF evaluation process count
        self.worker_threads = 4  # Thread count for other operations

        # Create thread pool for other operations
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.worker_threads,
            thread_name_prefix=f"Server{server_id}-General"
        )

        print(f"[Server {self.server_id}] Multiprocess data locality optimization mode")
        print(f"[Server {self.server_id}] Cache batch size: {self.cache_batch_size}")
        print(f"[Server {self.server_id}] VDPF evaluation processes: {self.vdpf_processes}")
        print(f"[Server {self.server_id}] Exchange directory: {self.exchange_dir}")
        
    def _load_data(self):
        """Load vector-level secret shared data"""
        print(f"[Server {self.server_id}] Loading {self.dataset} data...")
        
        # Use Test-Trident path uniformly
        self.data_dir = f"~/trident/dataset/{self.dataset}/server_{self.server_id}"
        
        # Load node vector shares
        self.nodes_path = os.path.join(self.data_dir, "nodes_shares.npy")
        self.node_shares = np.load(self.nodes_path)
        print(f"  Node data: {self.node_shares.shape}")
        print(f"  Data size: {self.node_shares.nbytes / 1024 / 1024:.1f}MB")
        
        print(f"  Available triples: {self.mult_server.triple_array.shape[0] - self.mult_server.used_count if self.mult_server.triple_array is not None else 0}")
        
    def start(self):
        """Start the server"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)
        
        print(f"[Server {self.server_id}] Listening on {self.host}:{self.port}")
        
        # Clean up old exchange files
        self._cleanup_exchange_files()
        
        try:
            while True:
                client_socket, address = server_socket.accept()
                print(f"[Server {self.server_id}] Accepted connection from {address}")
                
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket,)
                )
                client_thread.start()
                
        except KeyboardInterrupt:
            print(f"\n[Server {self.server_id}] Shutting down...")
        finally:
            server_socket.close()
            self._cleanup_exchange_files()
            self.executor.shutdown(wait=True)
            print(f"[Server {self.server_id}] Server closed")
    
    def _cleanup_exchange_files(self):
        """Clean up exchange files"""
        pattern = f"server_{self.server_id}_"
        current_time = time.time()
        for filename in os.listdir(self.exchange_dir):
            if filename.startswith(pattern):
                filepath = os.path.join(self.exchange_dir, filename)
                try:
                    # Clean up old files older than 5 minutes
                    if os.path.getmtime(filepath) < current_time - 300:
                        os.remove(filepath)
                except:
                    pass
    
    def _handle_client(self, client_socket: socket.socket):
        """Handle client requests"""
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
                print(f"[Server {self.server_id}] Received request: {request.get('command', 'unknown')}")
                
                response = self._process_request(request)
                print(f"[Server {self.server_id}] Request processing complete, response status: {response.get('status', 'unknown')}")
                
                response_data = json.dumps(response).encode()
                client_socket.sendall(len(response_data).to_bytes(4, 'big'))
                client_socket.sendall(response_data)
                
        except Exception as e:
            print(f"[Server {self.server_id}] Client handling error: {e}")
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
        else:
            return {'status': 'error', 'message': f'Unknown command: {command}'}
    
    def _multiprocess_vdpf_evaluation(self, serialized_key, num_nodes, num_batches):
        """Multiprocess VDPF evaluation - core optimization function (maintains original implementation)"""
        
        # Distribute batches evenly to processes
        batches_per_process = max(1, num_batches // self.vdpf_processes)
        remaining_batches = num_batches % self.vdpf_processes
        
        # Prepare process arguments
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
                    self.nodes_path,  # Pass file path instead of data
                    self.dataset  # Pass dataset name
                )
                process_args.append(args)
                current_batch = end_batch
        
        # Execute in parallel using process pool
        print(f"[Server {self.server_id}] Starting {len(process_args)} processes for VDPF evaluation")
        
        with Pool(processes=self.vdpf_processes) as pool:
            results = pool.map(evaluate_batch_range_process, process_args)
        
        # Merge results from all processes
        all_selector_shares = {}
        all_vector_shares = {}
        
        for process_result in results:
            all_selector_shares.update(process_result['selector_shares'])
            all_vector_shares.update(process_result['vector_shares'])
        
        print(f"[Server {self.server_id}] Multiprocess VDPF evaluation complete, evaluated {len(all_selector_shares)} nodes")
        
        return all_selector_shares, all_vector_shares
    
    def _save_exchange_data(self, query_id: str, e_shares: np.ndarray, f_shares: np.ndarray):
        """Save exchange data to file (uncompressed optimization)"""
        
        # Use .npz format without compression for faster I/O
        filename = f"server_{self.server_id}_query_{query_id}_data.npz"
        filepath = os.path.join(self.exchange_dir, filename)
        
        # Keep checksum calculation (for data integrity verification)
        data_hash = hashlib.md5((e_shares.tobytes() + f_shares.tobytes())).hexdigest()
        
        np.savez(filepath, 
                 e_shares=e_shares, 
                 f_shares=f_shares,
                 hash=data_hash)
        
        print(f"[Server {self.server_id}] Saved exchange data to {filename} (uncompressed optimization)")
    
    def _load_other_servers_data(self, query_id: str, num_nodes: int) -> Tuple[Dict, Dict]:
        """Load data from other servers"""
        
        all_e_from_others = {}
        all_f_from_others = {}
        
        for other_id in [1, 2, 3]:
            if other_id == self.server_id:
                continue
            
            filename = f"server_{other_id}_query_{query_id}_data.npz"
            filepath = os.path.join(self.exchange_dir, filename)
            
            # Wait for file to appear (maintains original wait mechanism)
            max_wait = 30
            for i in range(max_wait):
                if os.path.exists(filepath):
                    break
                time.sleep(1)
            
            if os.path.exists(filepath):
                # Load uncompressed .npz file (same format, just uncompressed)
                data = np.load(filepath)
                all_e_from_others[other_id] = data['e_shares']
                all_f_from_others[other_id] = data['f_shares']
                print(f"[Server {self.server_id}] Loaded data from Server {other_id} (uncompressed optimization)")
                
                # Optional: verify data integrity
                if 'hash' in data:
                    loaded_hash = str(data['hash'])
                    expected_hash = hashlib.md5((data['e_shares'].tobytes() + data['f_shares'].tobytes())).hexdigest()
                    if loaded_hash != expected_hash:
                        print(f"[Server {self.server_id}] Warning: Server {other_id} data checksum mismatch")
            else:
                print(f"[Server {self.server_id}] Warning: Unable to find data from Server {other_id}")
        
        return all_e_from_others, all_f_from_others
    
    def _handle_vector_node_query(self, request: Dict) -> Dict:
        """Vector-level node query - multiprocess data locality optimization version"""
        try:
            serialized_key = request['dpf_key']
            query_id = request.get('query_id', 'unknown')
            
            print(f"[Server {self.server_id}] Processing vector-level node query (multiprocess data locality optimization)")
            
            # 1. Deserialize key (perform in main process, verify key format)
            key = self.dpf_wrapper._deserialize_key(serialized_key)
            
            # 2. Initialize
            start_time = time.time()
            num_nodes = len(self.node_shares)
            vector_dim = self.node_shares.shape[1]  Dynamically get vector dimension
            result_accumulator = np.zeros(vector_dim, dtype=np.int64)
            
            print(f"[Server {self.server_id}] Total nodes: {num_nodes}, cache batch size: {self.cache_batch_size}")
            print(f"[Server {self.server_id}] DEBUG: Vector dimension: {vector_dim}, query ID: {query_id}")
            
            # 3. Phase 1: Multiprocess VDPF evaluation (keep original implementation)
            print(f"[Server {self.server_id}] Phase 1: Multiprocess VDPF evaluation ({self.vdpf_processes} processes)...")
            phase1_start = time.time()
            
            num_batches = (num_nodes + self.cache_batch_size - 1) // self.cache_batch_size
            print(f"[Server {self.server_id}] Will process {num_batches} batches using {self.vdpf_processes} processes")
            
            # Use multiprocess VDPF evaluation
            all_selector_shares, all_vector_shares = self._multiprocess_vdpf_evaluation(
                serialized_key, num_nodes, num_batches)
            
            phase1_time = time.time() - phase1_start
            total_ops_per_sec = num_nodes / phase1_time if phase1_time > 0 else 0
            print(f"[Server {self.server_id}] Phase 1 complete (multiprocess), elapsed {phase1_time:.2f}s, average {total_ops_per_sec:.0f} ops/sec")
            
            # DEBUG: Check VDPF selector values
            non_zero_count = sum(1 for v in all_selector_shares.values() if v != 0)
            print(f"[Server {self.server_id}] DEBUG: VDPF non-zero positions: {non_zero_count}")
            if non_zero_count > 0:
                for idx, val in list(all_selector_shares.items())[:5]:  # Print first 5 non-zero values
                    if val != 0:
                        print(f"[Server {self.server_id}] DEBUG: VDPF value at position {idx}: {val}")
            
            # 3.5. File sync barrier
            print(f"[Server {self.server_id}] Using file system synchronization...")
            self._file_sync_barrier(query_id, "phase1")
            
            # 4. Phase 2: Data locality optimization e/f computation (maintain original implementation)
            print(f"[Server {self.server_id}] Phase 2: Cache-friendly e/f computation...")
            phase2_start = time.time()
            
            all_e_shares = np.zeros(num_nodes, dtype=np.uint64)
            all_f_shares = np.zeros((num_nodes, vector_dim), dtype=np.uint64)
            all_computation_states = {}
            
            # Also use batch processing to avoid random memory access
            for batch_idx in range(num_batches):
                batch_start = batch_idx * self.cache_batch_size
                batch_end = min(batch_start + self.cache_batch_size, num_nodes)
                batch_size = batch_end - batch_start
                
                print(f"[Server {self.server_id}] e/f computation batch {batch_idx+1}/{num_batches}")
                
                # Bulk fetch triples (reduce lock contention)
                batch_triples = []
                for _ in range(batch_size):
                    batch_triples.append(self.mult_server.get_next_triple())
                
                # Process this batch sequentially
                for local_idx in range(batch_size):
                    global_idx = batch_start + local_idx
                    computation_id = f"query_{query_id}_pos{global_idx}"
                    
                    a, b, c = batch_triples[local_idx]
                    
                    e_share = (all_selector_shares[global_idx] - a) % self.field_size
                    all_e_shares[global_idx] = e_share
                    
                    # Vectorized computation of f values for all dimensions
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
            print(f"[Server {self.server_id}] Phase 2 complete, elapsed {phase2_time:.2f}s")

            # 5. Phase 3: Simulated batch exchange (maintain uncompressed optimization)
            print(f"[Server {self.server_id}] Phase 3: Simulated batch exchange...")
            phase3_start = time.time()

            # Save this server's data
            self._save_exchange_data(query_id, all_e_shares, all_f_shares)

            # Wait for other servers to finish saving
            self._file_sync_barrier(query_id, "phase3_save")

            # Load data from other servers
            all_e_from_others, all_f_from_others = self._load_other_servers_data(query_id, num_nodes)

            phase3_time = time.time() - phase3_start
            print(f"[Server {self.server_id}] Phase 3 complete (simulated), elapsed {phase3_time:.2f}s")
            
            # 6. Phase 4: Data locality optimization reconstruction computation (maintain NumPy parallelization optimization)
            print(f"[Server {self.server_id}] Phase 4: Cache-friendly reconstruction computation...")
            phase4_start = time.time()
            
            # Precompute Lagrange coefficients for all batches
            lagrange_1 = 2
            lagrange_2 = self.field_size - 1
            
            # Use batch processing for parallelized reconstruction computation
            for batch_idx in range(num_batches):
                batch_start = batch_idx * self.cache_batch_size
                batch_end = min(batch_start + self.cache_batch_size, num_nodes)
                batch_size = batch_end - batch_start
                
                print(f"[Server {self.server_id}] NumPy parallelized reconstruction batch {batch_idx+1}/{num_batches} ({batch_size} nodes)")
                
                # 1. Bulk extract data - avoid per-node access
                batch_e_shares_local = all_e_shares[batch_start:batch_end]  # shape: (batch_size,)
                batch_f_shares_local = all_f_shares[batch_start:batch_end, :]  # shape: (batch_size, vector_dim)
                
                # 2. Build share matrix for three servers
                # e_shares_matrix: shape (batch_size, 3)
                e_shares_matrix = np.zeros((batch_size, 3), dtype=np.uint64)
                e_shares_matrix[:, self.server_id - 1] = batch_e_shares_local
                
                # f_shares_matrix: shape (batch_size, vector_dim, 3)
                f_shares_matrix = np.zeros((batch_size, vector_dim, 3), dtype=np.uint64)
                f_shares_matrix[:, :, self.server_id - 1] = batch_f_shares_local
                
                # 3. Fill in data from other servers
                for other_id, other_e_shares in all_e_from_others.items():
                    e_shares_matrix[:, other_id - 1] = other_e_shares[batch_start:batch_end]
                
                for other_id, other_f_shares in all_f_from_others.items():
                    f_shares_matrix[:, :, other_id - 1] = other_f_shares[batch_start:batch_end, :]
                
                # 4. Vectorized e value reconstruction - process entire batch
                # Use Lagrange interpolation: e = e1 * 2 + e2 * (-1)
                batch_e_reconstructed = (e_shares_matrix[:, 0] * lagrange_1 + 
                                       e_shares_matrix[:, 1] * lagrange_2) % self.field_size
                # shape: (batch_size,)
                
                # 5. Vectorized f value reconstruction - process all dimensions
                # f_reconstructed: shape (batch_size, vector_dim)
                batch_f_reconstructed = (f_shares_matrix[:, :, 0] * lagrange_1 + 
                                       f_shares_matrix[:, :, 1] * lagrange_2) % self.field_size
                
                # 6. Bulk fetch triple data
                batch_triples = []
                for local_idx in range(batch_size):
                    global_idx = batch_start + local_idx
                    state = all_computation_states[global_idx]
                    a, b, c = state['triple']
                    batch_triples.append((a, b, c))
                
                # Convert to NumPy array for vectorization
                batch_a = np.array([t[0] for t in batch_triples], dtype=np.uint64)  # shape: (batch_size,)
                batch_b = np.array([t[1] for t in batch_triples], dtype=np.uint64)  # shape: (batch_size,)
                batch_c = np.array([t[2] for t in batch_triples], dtype=np.uint64)  # shape: (batch_size,)
                
                # 7. Vectorized final result computation - process all nodes and dimensions
                # Use broadcasting to compute batch_size × vector_dim result matrix
                
                # Expand dimensions to support broadcasting: (batch_size, 1) broadcasts to (batch_size, vector_dim)
                batch_e_expanded = batch_e_reconstructed[:, np.newaxis]  # shape: (batch_size, 1)
                batch_a_expanded = batch_a[:, np.newaxis]  # shape: (batch_size, 1)
                batch_b_expanded = batch_b[:, np.newaxis]  # shape: (batch_size, 1)
                batch_c_expanded = batch_c[:, np.newaxis]  # shape: (batch_size, 1)
                
                # Compute result: result = c + e*b + f*a + e*f (mod field_size)
                batch_result = batch_c_expanded  # shape: (batch_size, vector_dim)
                batch_result = (batch_result + batch_e_expanded * batch_b_expanded) % self.field_size
                batch_result = (batch_result + batch_f_reconstructed * batch_a_expanded) % self.field_size
                batch_result = (batch_result + batch_e_expanded * batch_f_reconstructed) % self.field_size
                
                # 8. Accumulate to total result - sum result for all nodes across vector dimensions
                batch_contribution = np.sum(batch_result, axis=0) % self.field_size  # shape: (vector_dim,)
                result_accumulator = (result_accumulator + batch_contribution) % self.field_size
                
                # DEBUG: Debug info for first batch
                if batch_idx == 0:
                    print(f"[Server {self.server_id}] DEBUG: First 5 values from batch 1: {batch_contribution[:5]}")
                    print(f"[Server {self.server_id}] DEBUG: First 5 accumulator values: {result_accumulator[:5]}")
                
                # 9. Clean up computation cache
                for local_idx in range(batch_size):
                    global_idx = batch_start + local_idx
                    state = all_computation_states[global_idx]
                    computation_id = state['computation_id']
                    if computation_id in self.mult_server.computation_cache:
                        del self.mult_server.computation_cache[computation_id]
                
                print(f"[Server {self.server_id}] Batch {batch_idx+1} parallelized reconstruction complete")
            
            phase4_time = time.time() - phase4_start
            total_time = time.time() - start_time
            
            print(f"[Server {self.server_id}] Query complete (multiprocess data locality optimization):")
            print(f"  Phase 1 (Multiprocess VDPF): {phase1_time:.2f}s")
            print(f"  Phase 2 (Cache-friendly e/f computation): {phase2_time:.2f}s")
            print(f"  Phase 3 (Simulated exchange): {phase3_time:.2f}s") 
            print(f"  Phase 4 (Cache-friendly reconstruction): {phase4_time:.2f}s")
            print(f"  Total: {total_time:.2f}s")
            
            # Add completion sync to ensure all servers finished reading data
            self._file_sync_barrier(query_id, "phase4_complete")
            
            # Now can safely clean up files
            self._cleanup_query_files(query_id)
            
            # Return result
            print(f"[Server {self.server_id}] Preparing to construct response...")
            
            # Safe type conversion
            try:
                result_list = [int(x) % (2**32) for x in result_accumulator]
                print(f"[Server {self.server_id}] Result conversion successful, length: {len(result_list)}")
                print(f"[Server {self.server_id}] DEBUG: First 5 final result values: {result_list[:5]}")
                print(f"[Server {self.server_id}] DEBUG: Final result range: [{min(result_list)}, {max(result_list)}]")
            except Exception as e:
                print(f"[Server {self.server_id}] Result conversion failed: {e}")
                result_list = [0] * vector_dim  # Fallback result
            
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
            
            print(f"[Server {self.server_id}] Response data construction complete")
            return response
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}
    
    def _file_sync_barrier(self, query_id: str, phase: str):
        """Implement synchronization barrier using file system"""
        # Create marker file for this server
        marker_file = f"server_{self.server_id}_query_{query_id}_{phase}_ready"
        marker_path = os.path.join(self.exchange_dir, marker_file)
        
        with open(marker_path, 'w') as f:
            f.write(str(time.time()))
        
        # Wait for marker files from other servers
        for other_id in [1, 2, 3]:
            if other_id == self.server_id:
                continue
            
            other_marker = f"server_{other_id}_query_{query_id}_{phase}_ready"
            other_path = os.path.join(self.exchange_dir, other_marker)
            
            max_wait = 60
            for i in range(max_wait):
                if os.path.exists(other_path):
                    break
                time.sleep(0.1)  # Reduce polling interval
            
            if not os.path.exists(other_path):
                print(f"[Server {self.server_id}] Warning: Server {other_id} did not complete {phase}")
    
    def _cleanup_query_files(self, query_id: str):
        """Clean up files for specific query"""
        for filename in os.listdir(self.exchange_dir):
            if query_id in filename and filename.endswith('_data.npz'):
                # Only clean up data files, not sync marker files
                try:
                    os.remove(os.path.join(self.exchange_dir, filename))
                except:
                    pass
    
    def _get_status(self) -> Dict:
        """Get server status"""
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
    """Main function"""
    # Set multiprocess start method to 'spawn' to ensure child processes initialize correctly
    multiprocessing.set_start_method('spawn', force=True)
    
    # Set command-line arguments
    parser = argparse.ArgumentParser(description='Vector-level multiprocess optimization server')
    parser.add_argument('--server-id', type=int, required=True, choices=[1, 2, 3],
                        help='Server ID (1, 2, or 3)')
    parser.add_argument('--dataset', type=str, default='laion', choices=['laion', 'siftsmall', 'tripclick', 'ms_marco', 'nfcorpus'],
                        help='Dataset name (default: laion)')
    parser.add_argument('--vdpf-processes', type=int, default=4,
                        help='VDPF evaluation processes (default: 4, range: 1-16)')
    
    args = parser.parse_args()
    
    if args.vdpf_processes < 1 or args.vdpf_processes > 16:
        print("Error: vdpf_processes must be between 1-16")
        sys.exit(1)
    
    print(f"Starting server {args.server_id}, dataset: {args.dataset}, using {args.vdpf_processes} VDPF evaluation processes")
    server = MemoryMappedOptimizedServer(args.server_id, args.dataset, args.vdpf_processes)
    server.start()


if __name__ == "__main__":
    main()