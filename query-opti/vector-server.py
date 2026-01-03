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
from multiprocessesing import Pool, cpu_count, Lock
import multiprocessesing
import argparse
from multiprocessesing import shared_memory

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('~/trident/standardDPF')
sys.path.append('~/trident/src')
sys.path.append('~/trident/query-opti')
sys.path.append('~/trident/query-opti')  # add optimization directory

from dpf_wrapper import VDPFVectorWrapper
from binary_serializer import BinaryKeySerializer  # import binary serializer
from dpf_wrapper_optimized import OptimizedVDPFVectorWrapper  # import optimized wrapper
from binary_protocol import BinaryProtocol  # import binary protocol
from secure_multiplication import NumpyMultiplicationServer
from basic_functionalities import get_config, Share, MPC23SSS


# Global function for use in processes pool calls (must be defined at module top level)
def warmup_processes(processes_id):
    """Warm up processes and load required modules"""
    import time
    import sys
    sys.path.append('~/trident/query-opti')
    # import required modules
    from dpf_wrapper_optimized import OptimizedVDPFVectorWrapper
    from binary_serializer import BinaryKeySerializer
    return processes_id

def evaluate_batch_range_processes(args):
    """Process pool worker function: evaluate VDPF for specified batch range"""
    processes_id, start_batch, end_batch, cache_batch_size, num_nodes, serialized_key, server_id, shm_name, shape, dtype, dataset_name = args

    # CPU affinity binding
    try:
        import sys
        sys.path.append('~/trident/query-opti')
        from cpu_affinity_optimizer import set_processes_affinity
        # Dynamically calculate cores each server should use
        total_cores = 16  # AMD EPYC 7R32 physical cores
        num_servers = 3   # Total of 3 servers
        cores_per_server = total_cores // num_servers  # Cores allocated per server

        # If processes count exceeds allocated cores, use round-robin allocation
        if processes_id < cores_per_server:
            set_processes_affinity(server_id, processes_id, cores_per_server)
        else:
            # Round-robin allocation: processes_id % cores_per_server
            effective_processes_id = processes_id % cores_per_server
            print(f"[Process {processes_id}] Warning: processes count ({processes_id+1}) exceeds core count ({cores_per_server}), binding to cores in round-robin")
            set_processes_affinity(server_id, effective_processes_id, cores_per_server)
    except Exception as e:
        print(f"[Process {processes_id}] CPU binding failed: {e}")
        import traceback
        traceback.print_exc()
    
    processes_total_start = time.time()

    # Print current CPU affinity
    import os
    current_cpus = os.sched_getaffinity(0)
    print(f"[Process {processes_id}] Current CPU affinity: {sorted(current_cpus)}")

    # Calculate actual number of nodes to processes
    actual_nodes = 0
    for batch_idx in range(start_batch, end_batch):
        batch_start = batch_idx * cache_batch_size
        batch_end = min(batch_start + cache_batch_size, num_nodes)
        actual_nodes += (batch_end - batch_start)

    print(f"[Process {processes_id}] Start evaluating batches {start_batch}-{end_batch-1} (actual nodes: {actual_nodes}) at {time.strftime('%H:%M:%S.%f')[:-3]}")

    # Timing 1: VDPF instance creation
    t1 = time.time()
    from dpf_wrapper_optimized import OptimizedVDPFVectorWrapper
    dpf_wrapper = OptimizedVDPFVectorWrapper(dataset_name=dataset_name)
    vdpf_init_time = time.time() - t1

    # Timing 2: Key deserialization
    t2 = time.time()
    if isinstance(serialized_key, bytes):
        key = BinaryKeySerializer.deserialize_vdpf23_key(serialized_key)
    else:
        # Compatible with old pickle format
        key = dpf_wrapper._deserialize_key(serialized_key)
    deserialize_time = time.time() - t2

    # Timing 3: Connect to shared memory
    t3 = time.time()
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    node_shares = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    shm_connect_time = time.time() - t3
    
    local_selector_shares = {}
    local_vector_shares = {}
    
    # Timing 4: VDPF evaluation detailed timing
    vdpf_eval_time = 0
    data_copy_time = 0
    batch_load_time = 0
    
    # VDPF internal time breakdown
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
        
        # Timing 4a: batch data load
        t4a = time.time()
        batch_data = node_shares[batch_start:batch_end].copy()
        batch_load_time += time.time() - t4a
        
        # Timing 4b: batch VDPF evaluation (with detailed measurement)
        t4b = time.time()
        
        # Temporarily modify eval_batch to collect internal timing
        batch_results = dpf_wrapper.eval_batch(key, batch_start, batch_end, server_id)
        
        vdpf_eval_time += time.time() - t4b
        
        # Timing 4c: result handling and data copying
        t4c = time.time()
        for local_idx in range(batch_size):
            global_idx = batch_start + local_idx
            local_selector_shares[global_idx] = batch_results[global_idx]
            local_vector_shares[global_idx] = batch_data[local_idx]
        data_copy_time += time.time() - t4c
    
    processes_total_time = time.time() - processes_total_start
    # Use previously calculated actual_nodes for accuracy
    evaluated_nodes = actual_nodes
    
    # Detailed timing analysis
    print(f"\n[Process {processes_id}] Detailed timing analysis (completed at {time.strftime('%H:%M:%S.%f')[:-3]}):")
    print(f"  - VDPF instance creation: {vdpf_init_time*1000:.1f}ms")
    print(f"  - key deserialization: {deserialize_time*1000:.1f}ms")
    print(f"  - shared memory connection: {shm_connect_time*1000:.1f}ms")
    print(f"  - batch data load: {batch_load_time*1000:.1f}ms ({batch_load_time/processes_total_time*100:.1f}%)")
    print(f"  - VDPF evaluation calculation: {vdpf_eval_time*1000:.1f}ms ({vdpf_eval_time/processes_total_time*100:.1f}%)")
    print(f"  - data copy operations: {data_copy_time*1000:.1f}ms ({data_copy_time/processes_total_time*100:.1f}%)")
    print(f"  - total time: {processes_total_time*1000:.1f}ms")
    print(f"  - nodes: {evaluated_nodes}, speed: {evaluated_nodes/processes_total_time:.0f} ops/sec")
    
    # Print cache statistics（ifcan/possibleuse）
    try:
        if hasattr(dpf_wrapper.vdpf, 'get_cache_stats'):
            cache_stats = dpf_wrapper.vdpf.get_cache_stats()
            print(f"  - PRG cache hit rate: {cache_stats['hit_rate']:.1f}% (hits:{cache_stats['total_hits']}, misses:{cache_stats['total_misses']})")
    except:
        pass
    
    # Close shared memory connection
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
            'total': processes_total_time
        }
    }


class MemoryMappedOptimizedServer:
    """Multiprocesses data locality optimization server（supports configurable processes count）"""
    
    def __init__(self, server_id: int, dataset: str = "laion", vdpf_processeses: int = 4):
        self.server_id = server_id
        self.dataset = dataset
        self.config = get_config(dataset)
        self.field_size = self.config.prime
        self.mpc = MPC23SSS(self.config)
        
        # Network configuration
        self.host = f"192.168.50.2{server_id}"
        self.port = 8000 + server_id
        
        # Initialize components
        self.dpf_wrapper = OptimizedVDPFVectorWrapper(dataset_name=dataset)
        self.mult_server = NumpyMultiplicationServer(server_id, self.config)
        
        # Load data
        self._load_data()
        
        # Exchange directory for simulation（Fall back to standard file system）
        self.exchange_dir = "/tmp/mpc_exchange"
        os.makedirs(self.exchange_dir, exist_ok=True)
        
        # Clean up old synchronization files
        try:
            for filename in os.listdir(self.exchange_dir):
                if f"server_{self.server_id}_" in filename:
                    try:
                        os.remove(os.path.join(self.exchange_dir, filename))
                    except:
                        pass
            print(f"[Server {self.server_id}] Cleaned up old synchronization files")
        except:
            pass
        
        # ===== Multiprocesses optimization parameters =====
        self.cache_batch_size = 1000  # Smaller batches for better load balancing
        self.vdpf_processeses = vdpf_processeses  # Configurable VDPF evaluation processes count
        self.worker_threads = 4  # Thread count for other operations
        
        # Create thread pool for other operations
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.worker_threads,
            thread_name_prefix=f"Server{server_id}-General"
        )
        
        # Create persistent processes pool to avoid creation overhead for each query
        self.processes_pool = Pool(processeses=self.vdpf_processeses)
        
        print(f"[Server {self.server_id}] multiprocessesData locality optimizationpattern")
        print(f"[Server {self.server_id}] Cache batch size: {self.cache_batch_size}")
        print(f"[Server {self.server_id}] VDPF evaluation processes count: {self.vdpf_processeses}")
        print(f"[Server {self.server_id}] Exchange directory: {self.exchange_dir}")
        print(f"[Server {self.server_id}] Process pool created（{self.vdpf_processeses}processes）")
        
        # Warm up processes pool
        self._warmup_processes_pool()
        
    def _warmup_processes_pool(self):
        """Warm up processes pool，Ensure all processeses have started and loaded required modules"""
        print(f"[Server {self.server_id}] Warm up processes pool...")
        
        # Execute warmup taskss
        warmup_start = time.time()
        results = self.processes_pool.map(warmup_processes, range(self.vdpf_processeses))
        warmup_time = time.time() - warmup_start
        
        print(f"[Server {self.server_id}] Process pool warmup completed, time taken {warmup_time:.2f}s")
    
    def _load_data(self):
        """Load vector-level secret shared data"""
        print(f"[Server {self.server_id}] load{self.dataset}data...")
        
        # Use unified Test-Trident path
        self.data_dir = f"~/trident/dataset/{self.dataset}/server_{self.server_id}"
        
        # Load node vector shares
        self.nodes_path = os.path.join(self.data_dir, "nodes_shares.npy")
        self.node_shares = np.load(self.nodes_path)
        print(f"  Node data: {self.node_shares.shape}")
        print(f"  Data size: {self.node_shares.nbytes / 1024 / 1024:.1f}MB")
        
        print(f"  Triples available: {self.mult_server.triple_array.shape[0] - self.mult_server.used_count if self.mult_server.triple_array is not None else 0}")
        
    def start(self):
        """Start server"""
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
            print(f"\n[Server {self.server_id}] Shutting down server...")
        finally:
            server_socket.close()
            self._cleanup_exchange_files()
            self.executor.shutdown(wait=True)
            # Shutting down processes pool
            if hasattr(self, 'processes_pool'):
                print(f"[Server {self.server_id}] Shutting down processes pool...")
                self.processes_pool.close()
                self.processes_pool.join()
            
            print(f"[Server {self.server_id}] Server has shut down")
    
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
        """Handle client request"""
        try:
            while True:
                # First try to read 4 bytes to determine if it is binary protocol
                length_bytes = client_socket.recv(4)
                if not length_bytes:
                    break
                
                length = int.from_bytes(length_bytes, 'big')
                
                # Check if it can be binary protocol（by checking length）
                if length < 1000000:  # reasonable request size
                    # Read first byte to check if it is command byte
                    first_byte = client_socket.recv(1)
                    if first_byte and first_byte[0] in [BinaryProtocol.CMD_QUERY_NODE_VECTOR, BinaryProtocol.CMD_GET_STATUS]:
                        # Is binary protocol
                        remaining = length - 1
                        data = first_byte
                        while len(data) < length:
                            chunk = client_socket.recv(min(remaining, 4096))
                            if not chunk:
                                break
                            data += chunk
                            remaining -= len(chunk)
                        
                        request = BinaryProtocol.decode_request(data)
                        print(f"[Server {self.server_id}] receivetobinaryrequest: {request.get('command', 'unknown')}")
                    else:
                        # JSON protocol
                        data = first_byte
                        while len(data) < length:
                            chunk = client_socket.recv(min(length - len(data), 4096))
                            if not chunk:
                                break
                            data += chunk
                        
                        request = json.loads(data.decode())
                        print(f"[Server {self.server_id}] receivedJSONrequest: {request.get('command', 'unknown')}")
                else:
                    # Length exception, skip
                    continue
                
                response = self._processes_request(request)
                print(f"[Server {self.server_id}] Request handling completed, response status: {response.get('status', 'unknown')}")
                
                # Encode response using binary protocol（Maintain original structure）
                response_data = BinaryProtocol.encode_response(response)
                client_socket.sendall(response_data)
                
        except Exception as e:
            print(f"[Server {self.server_id}] Error handling client: {e}")
            import traceback
            traceback.print_exc()
        finally:
            client_socket.close()
    
    def _processes_request(self, request: Dict) -> Dict:
        """Process request"""
        command = request.get('command')
        
        if command == 'query_node_vector':
            return self._handle_vector_node_query(request)
        elif command == 'get_status':
            return self._get_status()
        else:
            return {'status': 'error', 'message': f'Unknown command: {command}'}
    
    def _multiprocesses_vdpf_evaluation(self, serialized_key, num_nodes, num_batches):
        """Multiprocesses VDPF evaluation - use shared memory optimization"""
        
        # Create shared memory to avoid redundant I/O
        shm_start = time.time()
        print(f"[Server {self.server_id}] createsharememory...")
        shm = shared_memory.SharedMemory(create=True, size=self.node_shares.nbytes)
        shared_array = np.ndarray(self.node_shares.shape, dtype=self.node_shares.dtype, buffer=shm.buf)
        shared_array[:] = self.node_shares[:]
        shm_time = time.time() - shm_start
        print(f"[Server {self.server_id}] Shared memory creation time: {shm_time:.3f}s")
        
        # Improved load balancing allocation algorithm
        # Calculate number of nodes each processes should handle, not batch count
        nodes_per_processes = num_nodes // self.vdpf_processeses
        remaining_nodes = num_nodes % self.vdpf_processeses
        
        # Prepare processes parameters
        processes_args = []
        current_node_start = 0
        
        for processes_id in range(self.vdpf_processeses):
            # Calculate number of nodes this processes should handle
            processes_nodes = nodes_per_processes + (1 if processes_id < remaining_nodes else 0)
            
            if processes_nodes == 0:
                continue
                
            # Calculate start and end nodes for this processes
            node_start = current_node_start
            node_end = node_start + processes_nodes
            
            # Convert to batch index
            start_batch = node_start // self.cache_batch_size
            end_batch = (node_end + self.cache_batch_size - 1) // self.cache_batch_size
            
            args = (
                processes_id,
                start_batch,
                min(end_batch, num_batches),
                self.cache_batch_size,
                num_nodes,
                serialized_key,
                self.server_id,
                shm.name,  # Pass shared memory name
                self.node_shares.shape,  # Pass array shape
                self.node_shares.dtype,  # Pass data type
                self.dataset  # Pass dataset name
            )
            processes_args.append(args)
            current_node_start = node_end
            
        # Print load distribution information
        print(f"[Server {self.server_id}] Load balancing allocation:")
        actual_node_counts = []
        for i, args in enumerate(processes_args):
            processes_id, start_batch, end_batch = args[0:3]
            batches = end_batch - start_batch
            # Calculate actual node count
            actual_nodes = 0
            for batch_idx in range(start_batch, end_batch):
                batch_start = batch_idx * self.cache_batch_size
                batch_end = min(batch_start + self.cache_batch_size, num_nodes)
                actual_nodes += (batch_end - batch_start)
            actual_node_counts.append(actual_nodes)
            print(f"  - Process {processes_id}: batches {start_batch}-{end_batch-1} ({batches} batches, {actual_nodes} node)")
        
        # Calculate load balancing statistics
        if actual_node_counts:
            avg_nodes = np.mean(actual_node_counts)
            std_nodes = np.std(actual_node_counts)
            max_nodes = max(actual_node_counts)
            min_nodes = min(actual_node_counts)
            print(f"[Server {self.server_id}] Load statistics: average={avg_nodes:.0f}, standarddeviation={std_nodes:.1f}, max={max_nodes}, min={min_nodes}")
        
        # Use persistent processes pool to execute in parallel
        print(f"[Server {self.server_id}] usepersistentprocessespoolexecuteVDPFevaluation（{len(processes_args)}tasks）")
        
        # Use already created processes pool
        results = self.processes_pool.map(evaluate_batch_range_processes, processes_args)
        
        # Merge results from all processeses
        all_selector_shares = {}
        all_vector_shares = {}
        processes_timings = []
        
        for processes_result in results:
            all_selector_shares.update(processes_result['selector_shares'])
            all_vector_shares.update(processes_result['vector_shares'])
            if 'timing' in processes_result:
                processes_timings.append(processes_result['timing'])
        
        # Aggregate timing analysis
        if processes_timings:
            print(f"\n[Server {self.server_id}] Phase 1 timing analysis aggregate:")
            avg_vdpf_init = np.mean([t['vdpf_init'] for t in processes_timings]) * 1000
            avg_deserialize = np.mean([t['deserialize'] for t in processes_timings]) * 1000
            avg_shm_connect = np.mean([t['shm_connect'] for t in processes_timings]) * 1000
            
            # Actual parallel execution time is the slowest processes time
            max_vdpf_eval = max([t['vdpf_eval'] for t in processes_timings]) * 1000
            max_total_time = max([t['total'] for t in processes_timings]) * 1000
            
            # Total CPU time（used to understand total workload）
            total_cpu_time = sum([t['total'] for t in processes_timings]) * 1000
            total_vdpf_cpu = sum([t['vdpf_eval'] for t in processes_timings]) * 1000
            
            print(f"  === Parallel execution analysis ===")
            print(f"  - Slowest processes total time: {max_total_time:.1f}ms (This is the actual parallel execution time)")
            print(f"  - Slowest processes VDPF time: {max_vdpf_eval:.1f}ms")
            print(f"  - Parallel efficiency: {total_cpu_time/max_total_time/self.vdpf_processeses*100:.1f}%")
            print(f"  ")
            print(f"  === Overhead analysis ===")
            print(f"  - Average VDPF instance creation: {avg_vdpf_init:.1f}ms")
            print(f"  - Average key deserialization: {avg_deserialize:.1f}ms")
            print(f"  - Average shared memory connection: {avg_shm_connect:.1f}ms")
            print(f"  - Shared memory creation: {shm_time*1000:.1f}ms")
            print(f"  ")
            print(f"  === CPU time statistics ===")
            print(f"  - Total CPU time for all processeses: {total_cpu_time:.1f}ms")
            print(f"  - Total VDPF calculation time for all processeses: {total_vdpf_cpu:.1f}ms")
        
        print(f"[Server {self.server_id}] Multiprocesses VDPF evaluation completed, total evaluated {len(all_selector_shares)} node")
        
        # Clean up shared memory
        shm.close()
        shm.unlink()
        
        return all_selector_shares, all_vector_shares
    
    def _save_exchange_data(self, query_id: str, e_shares: np.ndarray, f_shares: np.ndarray):
        """Save exchange data to file（No compression optimization）"""
        
        # Use .npz format without compression to speed up I/O
        filename = f"server_{self.server_id}_query_{query_id}_data.npz"
        filepath = os.path.join(self.exchange_dir, filename)
        
        # Keep checksum calculation（for data integrity verification）
        data_hash = hashlib.md5((e_shares.tobytes() + f_shares.tobytes())).hexdigest()
        
        np.savez(filepath, 
                 e_shares=e_shares, 
                 f_shares=f_shares,
                 hash=data_hash)
        
        # print(f"[Server {self.server_id}] Saved exchange data to {filename}（No compression optimization）")
    
    def _load_other_servers_data(self, query_id: str, num_nodes: int) -> Tuple[Dict, Dict]:
        """Load data from other servers"""
        
        all_e_from_others = {}
        all_f_from_others = {}
        
        for other_id in [1, 2, 3]:
            if other_id == self.server_id:
                continue
            
            filename = f"server_{other_id}_query_{query_id}_data.npz"
            filepath = os.path.join(self.exchange_dir, filename)
            
            # Wait for file to appear（Maintain original wait mechanism）
            max_wait = 30
            for i in range(max_wait):
                if os.path.exists(filepath):
                    break
                time.sleep(1)
            
            if os.path.exists(filepath):
                # Load uncompressed .npz file（Same format, just without compression）
                data = np.load(filepath)
                all_e_from_others[other_id] = data['e_shares']
                all_f_from_others[other_id] = data['f_shares']
                # print(f"[Server {self.server_id}] alreadyload Server {other_id} data（No compression optimization）")
                
                # Optional: validate data integrity
                if 'hash' in data:
                    loaded_hash = str(data['hash'])
                    expected_hash = hashlib.md5((data['e_shares'].tobytes() + data['f_shares'].tobytes())).hexdigest()
                    if loaded_hash != expected_hash:
                        pass  # print(f"[Server {self.server_id}] warning: Server {other_id} datachecksumandnotmatch")
            else:
                pass  # print(f"[Server {self.server_id}] warning: cannot findto Server {other_id} data")
        
        return all_e_from_others, all_f_from_others
    
    def _handle_vector_node_query(self, request: Dict) -> Dict:
        """Vector-level node query - multiprocesses data locality optimization version"""
        try:
            serialized_key = request['dpf_key']
            query_id = request.get('query_id', 'unknown')
            
            print(f"[Server {self.server_id}] Handle vector-level node query（multiprocessesData locality optimization）")
            
            # 1. Deserialize key（Performed in main processes to verify key format）
            key = self.dpf_wrapper._deserialize_key(serialized_key)
            
            # 2. Initialize
            start_time = time.time()
            num_nodes = len(self.node_shares)
            vector_dim = self.node_shares.shape[1]  # Dynamically get vector dimension
            result_accumulator = np.zeros(vector_dim, dtype=np.int64)
            
            print(f"[Server {self.server_id}] Total number of nodes: {num_nodes}, Cache batch size: {self.cache_batch_size}")
            # print(f"[Server {self.server_id}] DEBUG: vector dimension: {vector_dim}, query ID: {query_id}")
            
            # 3. Phase1：Multiprocesses VDPF evaluation（Maintain original implementation)
            print(f"[Server {self.server_id}] Phase1: Multiprocesses VDPF evaluation ({self.vdpf_processeses} processes)...")
            phase1_start = time.time()
            
            num_batches = (num_nodes + self.cache_batch_size - 1) // self.cache_batch_size
            print(f"[Server {self.server_id}] willprocesses/handle {num_batches} batches，use {self.vdpf_processeses} processes")
            
            # Use multiprocesses VDPF evaluation
            all_selector_shares, all_vector_shares = self._multiprocesses_vdpf_evaluation(
                serialized_key, num_nodes, num_batches)
            
            phase1_time = time.time() - phase1_start
            total_ops_per_sec = num_nodes / phase1_time if phase1_time > 0 else 0
            print(f"[Server {self.server_id}] Phase1completed（multiprocesses），time taken {phase1_time:.2f}s, average {total_ops_per_sec:.0f} ops/sec")
            
            # # DEBUG: check VDPF selector values
            # non_zero_count = sum(1 for v in all_selector_shares.values() if v != 0)
            # print(f"[Server {self.server_id}] DEBUG: VDPF non-zero positions: {non_zero_count}")
            # if non_zero_count > 0:
            #     for idx, val in list(all_selector_shares.items())[:5]:  # Print first 5 non-zero values
            #         if val != 0:
            #             print(f"[Server {self.server_id}] DEBUG: bitposition {idx} VDPFvalue: {val}")
            
            # 3.5. File synchronization barrier
            # print(f"[Server {self.server_id}] Using file system synchronization...")
            self._file_sync_barrier(query_id, "phase1")
            
            # 4. Phase2：Data locality optimized e/f calculation（Maintain original implementation）
            # print(f"[Server {self.server_id}] Phase2: Cache-friendly e/f calculation...")
            phase2_start = time.time()
            
            all_e_shares = np.zeros(num_nodes, dtype=np.uint64)
            all_f_shares = np.zeros((num_nodes, vector_dim), dtype=np.uint64)
            all_computation_states = {}
            
            # Also use batch processesing to avoid random memory access
            for batch_idx in range(num_batches):
                batch_start = batch_idx * self.cache_batch_size
                batch_end = min(batch_start + self.cache_batch_size, num_nodes)
                batch_size = batch_end - batch_start
                
                # print(f"[Server {self.server_id}] e/f calculation batch {batch_idx+1}/{num_batches}")
                
                # Batch get triples（reduce lock contention）
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
            # print(f"[Server {self.server_id}] Phase2completed，time taken {phase2_time:.2f}s")
            
            # 5. Phase3：Simulate batch exchange（Maintain no-compression optimization）
            # print(f"[Server {self.server_id}] Phase3: Simulate batch exchange...")
            phase3_start = time.time()
            
            # Save this server's data
            self._save_exchange_data(query_id, all_e_shares, all_f_shares)
            
            # Wait for other servers to complete saving
            self._file_sync_barrier(query_id, "phase3_save")
            
            # Read other servers' data
            all_e_from_others, all_f_from_others = self._load_other_servers_data(query_id, num_nodes)
            
            phase3_time = time.time() - phase3_start
            # print(f"[Server {self.server_id}] Phase3completed（simulated），time taken {phase3_time:.2f}s")
            
            # 6. Phase4：Data locality optimized reconstruction calculation（maintainNumPyparallelizedoptimization）
            # print(f"[Server {self.server_id}] Phase4: Cache-friendly reconstruction calculation...")
            phase4_start = time.time()
            
            # Pre-calculate Lagrange coefficients for all batches
            lagrange_1 = 2
            lagrange_2 = self.field_size - 1
            
            # Use batch processesing for parallelized reconstruction calculation
            for batch_idx in range(num_batches):
                batch_start = batch_idx * self.cache_batch_size
                batch_end = min(batch_start + self.cache_batch_size, num_nodes)
                batch_size = batch_end - batch_start
                
                # print(f"[Server {self.server_id}] NumPy parallelized reconstruction batch {batch_idx+1}/{num_batches} ({batch_size} node)")
                
                # 1. Batch extract data - avoid per-node access
                batch_e_shares_local = all_e_shares[batch_start:batch_end]  # shape: (batch_size,)
                batch_f_shares_local = all_f_shares[batch_start:batch_end, :]  # shape: (batch_size, vector_dim)
                
                # 2. Build three-server share matrix
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
                
                # 4. Vectorized e value reconstruction - batch processes entire batch
                # Using Lagrange interpolation: e = e1 * 2 + e2 * (-1)
                batch_e_reconstructed = (e_shares_matrix[:, 0] * lagrange_1 + 
                                       e_shares_matrix[:, 1] * lagrange_2) % self.field_size
                # shape: (batch_size,)
                
                # 5. Vectorized f value reconstruction - batch processes all dimensions
                # f_reconstructed: shape (batch_size, vector_dim)
                batch_f_reconstructed = (f_shares_matrix[:, :, 0] * lagrange_1 + 
                                       f_shares_matrix[:, :, 1] * lagrange_2) % self.field_size
                
                # 6. Batch get triple data
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
                
                # 7. Vectorized computation of final result - batch processes all nodes and dimensions
                # Use broadcasting to calculate batch_size × vector_dim result matrix
                
                # Expand dimensions to support broadcasting: (batch_size, 1) broadcast to (batch_size, vector_dim)
                batch_e_expanded = batch_e_reconstructed[:, np.newaxis]  # shape: (batch_size, 1)
                batch_a_expanded = batch_a[:, np.newaxis]  # shape: (batch_size, 1)
                batch_b_expanded = batch_b[:, np.newaxis]  # shape: (batch_size, 1)
                batch_c_expanded = batch_c[:, np.newaxis]  # shape: (batch_size, 1)
                
                # Calculate result: result = c + e*b + f*a + e*f (mod field_size)
                batch_result = batch_c_expanded  # shape: (batch_size, vector_dim)
                batch_result = (batch_result + batch_e_expanded * batch_b_expanded) % self.field_size
                batch_result = (batch_result + batch_f_reconstructed * batch_a_expanded) % self.field_size
                batch_result = (batch_result + batch_e_expanded * batch_f_reconstructed) % self.field_size
                
                # 8. Accumulate to total result - sum vector dimension results for all nodes
                batch_contribution = np.sum(batch_result, axis=0) % self.field_size  # shape: (vector_dim,)
                result_accumulator = (result_accumulator + batch_contribution) % self.field_size
                
                # # DEBUG: Debug information for first batch
                # if batch_idx == 0:
                #     print(f"[Server {self.server_id}] DEBUG: First 5 values contributed by batch 1: {batch_contribution[:5]}")
                #     print(f"[Server {self.server_id}] DEBUG: First 5 values in accumulator: {result_accumulator[:5]}")
                
                # 9. Clean up calculation cache
                for local_idx in range(batch_size):
                    global_idx = batch_start + local_idx
                    state = all_computation_states[global_idx]
                    computation_id = state['computation_id']
                    if computation_id in self.mult_server.computation_cache:
                        del self.mult_server.computation_cache[computation_id]
                
                # print(f"[Server {self.server_id}] batches {batch_idx+1} parallelized reconstructioncompleted")
            
            phase4_time = time.time() - phase4_start
            total_time = time.time() - start_time
            
            print(f"[Server {self.server_id}] Query completed（multiprocessesData locality optimization）:")
            print(f"  Phase1 (multiprocessesVDPF): {phase1_time:.2f}s")
            print(f"  Phase2 (cache-friendly e/f calculation): {phase2_time:.2f}s")
            print(f"  Phase3 (simulated exchange): {phase3_time:.2f}s") 
            print(f"  Phase4 (cache-friendly reconstruction): {phase4_time:.2f}s")
            print(f"  total: {total_time:.2f}s")
            
            # Add completion synchronization to ensure all servers have read data
            self._file_sync_barrier(query_id, "phase4_complete")
            
            # Delayed cleanup to give other servers time to complete synchronization
            # Only clean up data files, synchronization files will be cleaned up at next query start
            self._cleanup_data_files_only(query_id)
            
            # Return result
            # print(f"[Server {self.server_id}] Prepare to construct response...")
            
            # Safe type conversion
            try:
                result_list = [int(x) % (2**32) for x in result_accumulator]
                # print(f"[Server {self.server_id}] Result conversion successful, length: {len(result_list)}")
                # print(f"[Server {self.server_id}] DEBUG: Final result first 5 values: {result_list[:5]}")
                # print(f"[Server {self.server_id}] DEBUG: Final result range: [{min(result_list)}, {max(result_list)}]")
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
                    'vdpf_processeses': self.vdpf_processeses,
                    'avg_ops_per_sec_phase1': total_ops_per_sec
                }
            }
            
            # print(f"[Server {self.server_id}] Response data construction completed")
            return response
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}
    
    def _file_sync_barrier(self, query_id: str, phase: str):
        """Use file system to implement synchronization barrier"""
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
                print(f"[Server {self.server_id}] warning: Server {other_id} notcompleted {phase}")
    
    def _cleanup_query_files(self, query_id: str):
        """Clean up files for specific query"""
        for filename in os.listdir(self.exchange_dir):
            if query_id in filename:
                # Clean up data files and synchronization marker files
                try:
                    os.remove(os.path.join(self.exchange_dir, filename))
                except:
                    pass
    
    def _cleanup_data_files_only(self, query_id: str):
        """Only clean up data files, keep synchronization files"""
        for filename in os.listdir(self.exchange_dir):
            if query_id in filename and filename.endswith('_data.npz'):
                # Only clean up data files
                try:
                    os.remove(os.path.join(self.exchange_dir, filename))
                except:
                    pass
    
    def _get_status(self) -> Dict:
        """Get server status"""
        return {
            'status': 'success',
            'server_id': self.server_id,
            'mode': 'multiprocesses_locality_optimized',
            'cache_batch_size': self.cache_batch_size,
            'vdpf_processeses': self.vdpf_processeses,
            'worker_threads': self.worker_threads,
            'data_loaded': {
                'nodes': self.node_shares.shape
            },
            'triples_available': self.mult_server.triple_array.shape[0] - self.mult_server.used_count if self.mult_server.triple_array is not None else 0,
            'triples_used': self.mult_server.used_count,
            'exchange_dir': self.exchange_dir
        }


def main():
    """mainfunction"""
    # setmultiprocessesstartmethodas'spawn'，ensure childprocessescorrectInitialize
    multiprocessesing.set_start_method('spawn', force=True)
    
    # setcommand lineparameter
    parser = argparse.ArgumentParser(description='vector-level multiprocessesoptimizationserver')
    parser.add_argument('--server-id', type=int, required=True, choices=[1, 2, 3],
                        help='serverID (1, 2, or 3)')
    parser.add_argument('--dataset', type=str, default='laion', choices=['laion', 'siftsmall', 'tripclick', 'ms_marco', 'nfcorpus'],
                        help='datadataset name (default: laion)')
    parser.add_argument('--vdpf-processeses', type=int, default=4,
                        help='VDPF evaluation processes count (default: 4, range: 1-16)')
    
    args = parser.parse_args()
    
    if args.vdpf_processeses < 1 or args.vdpf_processeses > 16:
        print("error: vdpf_processeses mustmustin 1-16 between")
        sys.exit(1)
    
    print(f"Start server {args.server_id}，dataset: {args.dataset}，use {args.vdpf_processeses} VDPFevaluationprocesses")
    server = MemoryMappedOptimizedServer(args.server_id, args.dataset, args.vdpf_processeses)
    server.start()


if __name__ == "__main__":
    main()