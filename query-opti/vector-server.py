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
sys.path.append('~/trident/query-opti')  # Add optimization directory

from dpf_wrapper import VDPFVectorWrapper
from binary_serializer import BinaryKeySerializer  # Import binary serializer
from dpf_wrapper_optimized import OptimizedVDPFVectorWrapper  # Import optimized wrapper
from binary_protocol import BinaryProtocol  # Import binary protocol
from secure_multiplication import NumpyMultiplicationServer
from basic_functionalities import get_config, Share, MPC23SSS


# Global function for process pool (must be defined at module top level)
def warmup_process(process_id):
    """Warm up process, load necessary modules"""
    import time
    import sys
    sys.path.append('~/trident/query-opti')
    # Import necessary modules
    from dpf_wrapper_optimized import OptimizedVDPFVectorWrapper
    from binary_serializer import BinaryKeySerializer
    return process_id

def evaluate_batch_range_process(args):
    """Process pool worker: Evaluate specified batch range of VDPF"""
    process_id, start_batch, end_batch, cache_batch_size, num_nodes, serialized_key, server_id, shm_name, shape, dtype, dataset_name = args
    
    # CPU affinity binding
    try:
        import sys
        sys.path.append('~/trident/query-opti')
        from cpu_affinity_optimizer import set_process_affinity
        # Dynamically calculate cores per server
        total_cores = 16  # AMD EPYC 7R32[CN]physical cores
        num_servers = 3   # total3servers
        cores_per_server = total_cores // num_servers  # cores allocated per server
        
        # If process count exceeds allocated cores, use round-robin allocation
        if process_id < cores_per_server:
            set_process_affinity(server_id, process_id, cores_per_server)
        else:
            # round-robin allocation：process_id % cores_per_server
            effective_process_id = process_id % cores_per_server
            print(f"[Process {process_id}] Warning: process count({process_id+1})exceeds core count({cores_per_server})，cyclically bind to core")
            set_process_affinity(server_id, effective_process_id, cores_per_server)
    except Exception as e:
        print(f"[Process {process_id}] CPU binding failed: {e}")
        import traceback
        traceback.print_exc()
    
    process_total_start = time.time()
    
    # printCurrent CPU affinity
    import os
    current_cpus = os.sched_getaffinity(0)
    print(f"[Process {process_id}] Current CPU affinity: {sorted(current_cpus)}")
    
    # Calculate actualprocessnumber of nodes
    actual_nodes = 0
    for batch_idx in range(start_batch, end_batch):
        batch_start = batch_idx * cache_batch_size
        batch_end = min(batch_start + cache_batch_size, num_nodes)
        actual_nodes += (batch_end - batch_start)
    
    print(f"[Process {process_id}] Start evaluating batch {start_batch}-{end_batch-1} ([CN]: {actual_nodes}) at {time.strftime('%H:%M:%S.%f')[:-3]}")
    
    # [CN]1：VDPFinstancecreate
    t1 = time.time()
    from dpf_wrapper_optimized import OptimizedVDPFVectorWrapper
    dpf_wrapper = OptimizedVDPFVectorWrapper(dataset_name=dataset_name)
    vdpf_init_time = time.time() - t1
    
    # [CN]2：[CN]
    t2 = time.time()
    if isinstance(serialized_key, bytes):
        key = BinaryKeySerializer.deserialize_vdpf23_key(serialized_key)
    else:
        # [CN]pickle[CN]
        key = dpf_wrapper._deserialize_key(serialized_key)
    deserialize_time = time.time() - t2
    
    # [CN]3：connect[CN]
    t3 = time.time()
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    node_shares = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    shm_connect_time = time.time() - t3
    
    local_selector_shares = {}
    local_vector_shares = {}
    
    # [CN]4：VDPF[CN]
    vdpf_eval_time = 0
    data_copy_time = 0
    batch_load_time = 0
    
    # VDPF[CN]
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
        
        # [CN]4a：[CN]
        t4a = time.time()
        batch_data = node_shares[batch_start:batch_end].copy()
        batch_load_time += time.time() - t4a
        
        # [CN]4b：[CN]VDPF[CN]（[CN]）
        t4b = time.time()
        
        # [CN]eval_batch[CN]
        batch_results = dpf_wrapper.eval_batch(key, batch_start, batch_end, server_id)
        
        vdpf_eval_time += time.time() - t4b
        
        # [CN]4c：[CN]process[CN]
        t4c = time.time()
        for local_idx in range(batch_size):
            global_idx = batch_start + local_idx
            local_selector_shares[global_idx] = batch_results[global_idx]
            local_vector_shares[global_idx] = batch_data[local_idx]
        data_copy_time += time.time() - t4c
    
    process_total_time = time.time() - process_total_start
    # [CN]calculate[CN]actual_nodes，[CN]
    evaluated_nodes = actual_nodes
    
    # [CN]
    print(f"\n[Process {process_id}] [CN] ([CN] {time.strftime('%H:%M:%S.%f')[:-3]}):")
    print(f"  - VDPFinstancecreate: {vdpf_init_time*1000:.1f}ms")
    print(f"  - [CN]: {deserialize_time*1000:.1f}ms")
    print(f"  - [CN]connect: {shm_connect_time*1000:.1f}ms")
    print(f"  - [CN]: {batch_load_time*1000:.1f}ms ({batch_load_time/process_total_time*100:.1f}%)")
    print(f"  - VDPF[CN]calculate: {vdpf_eval_time*1000:.1f}ms ({vdpf_eval_time/process_total_time*100:.1f}%)")
    print(f"  - [CN]: {data_copy_time*1000:.1f}ms ({data_copy_time/process_total_time*100:.1f}%)")
    print(f"  - [CN]: {process_total_time*1000:.1f}ms")
    print(f"  - [CN]: {evaluated_nodes}, [CN]: {evaluated_nodes/process_total_time:.0f} ops/sec")
    
    # print[CN]（[CN]）
    try:
        if hasattr(dpf_wrapper.vdpf, 'get_cache_stats'):
            cache_stats = dpf_wrapper.vdpf.get_cache_stats()
            print(f"  - PRG[CN]: {cache_stats['hit_rate']:.1f}% ([CN]:{cache_stats['total_hits']}, [CN]:{cache_stats['total_misses']})")
    except:
        pass
    
    # [CN]connect
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
    """[CN]（[CN]）"""
    
    def __init__(self, server_id: int, dataset: str = "laion", vdpf_processes: int = 4):
        self.server_id = server_id
        self.dataset = dataset
        self.config = get_config(dataset)
        self.field_size = self.config.prime
        self.mpc = MPC23SSS(self.config)
        
        # [CN]
        self.host = f"192.168.50.2{server_id}"
        self.port = 8000 + server_id
        
        # initialize[CN]
        self.dpf_wrapper = OptimizedVDPFVectorWrapper(dataset_name=dataset)
        self.mult_server = NumpyMultiplicationServer(server_id, self.config)
        
        # [CN]
        self._load_data()
        
        # [CN]（[CN]）
        self.exchange_dir = "/tmp/mpc_exchange"
        os.makedirs(self.exchange_dir, exist_ok=True)
        
        # [CN]
        try:
            for filename in os.listdir(self.exchange_dir):
                if f"server_{self.server_id}_" in filename:
                    try:
                        os.remove(os.path.join(self.exchange_dir, filename))
                    except:
                        pass
            print(f"[Server {self.server_id}] [CN]")
        except:
            pass
        
        # ===== [CN] =====
        self.cache_batch_size = 1000  # [CN]
        self.vdpf_processes = vdpf_processes  # [CN]VDPF[CN]
        self.worker_threads = 4  # [CN]
        
        # [CN]create[CN]
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.worker_threads,
            thread_name_prefix=f"Server{server_id}-General"
        )
        
        # create[CN]，[CN]create[CN]
        self.process_pool = Pool(processes=self.vdpf_processes)
        
        print(f"[Server {self.server_id}] [CN]")
        print(f"[Server {self.server_id}] [CN]: {self.cache_batch_size}")
        print(f"[Server {self.server_id}] VDPF[CN]: {self.vdpf_processes}")
        print(f"[Server {self.server_id}] [CN]: {self.exchange_dir}")
        print(f"[Server {self.server_id}] [CN]create（{self.vdpf_processes}[CN]）")
        
        # Warm up process[CN]
        self._warmup_process_pool()
        
    def _warmup_process_pool(self):
        """Warm up process[CN]，[CN]start[CN]load necessary modules"""
        print(f"[Server {self.server_id}] Warm up process[CN]...")
        
        # [CN]
        warmup_start = time.time()
        results = self.process_pool.map(warmup_process, range(self.vdpf_processes))
        warmup_time = time.time() - warmup_start
        
        print(f"[Server {self.server_id}] [CN]，[CN] {warmup_time:.2f}[CN]")
    
    def _load_data(self):
        """[CN]"""
        print(f"[Server {self.server_id}] [CN]{self.dataset}[CN]...")
        
        # [CN]Test-Trident[CN]
        self.data_dir = f"~/trident/dataset/{self.dataset}/server_{self.server_id}"
        
        # [CN]
        self.nodes_path = os.path.join(self.data_dir, "nodes_shares.npy")
        self.node_shares = np.load(self.nodes_path)
        print(f"  [CN]: {self.node_shares.shape}")
        print(f"  [CN]: {self.node_shares.nbytes / 1024 / 1024:.1f}MB")
        
        print(f"  [CN]: {self.mult_server.triple_array.shape[0] - self.mult_server.used_count if self.mult_server.triple_array is not None else 0}")
        
    def start(self):
        """start[CN]"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)
        
        print(f"[Server {self.server_id}] [CN] {self.host}:{self.port}")
        
        # [CN]
        self._cleanup_exchange_files()
        
        try:
            while True:
                client_socket, address = server_socket.accept()
                print(f"[Server {self.server_id}] [CN]connect[CN] {address}")
                
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket,)
                )
                client_thread.start()
                
        except KeyboardInterrupt:
            print(f"\n[Server {self.server_id}] [CN]...")
        finally:
            server_socket.close()
            self._cleanup_exchange_files()
            self.executor.shutdown(wait=True)
            # [CN]
            if hasattr(self, 'process_pool'):
                print(f"[Server {self.server_id}] [CN]...")
                self.process_pool.close()
                self.process_pool.join()
            
            print(f"[Server {self.server_id}] [CN]")
    
    def _cleanup_exchange_files(self):
        """[CN]"""
        pattern = f"server_{self.server_id}_"
        current_time = time.time()
        for filename in os.listdir(self.exchange_dir):
            if filename.startswith(pattern):
                filepath = os.path.join(self.exchange_dir, filename)
                try:
                    # [CN]5[CN]
                    if os.path.getmtime(filepath) < current_time - 300:
                        os.remove(filepath)
                except:
                    pass
    
    def _handle_client(self, client_socket: socket.socket):
        """process[CN]"""
        try:
            while True:
                # [CN]4[CN]
                length_bytes = client_socket.recv(4)
                if not length_bytes:
                    break
                
                length = int.from_bytes(length_bytes, 'big')
                
                # [CN]（[CN]）
                if length < 1000000:  # [CN]
                    # [CN]
                    first_byte = client_socket.recv(1)
                    if first_byte and first_byte[0] in [BinaryProtocol.CMD_QUERY_NODE_VECTOR, BinaryProtocol.CMD_GET_STATUS]:
                        # [CN]
                        remaining = length - 1
                        data = first_byte
                        while len(data) < length:
                            chunk = client_socket.recv(min(remaining, 4096))
                            if not chunk:
                                break
                            data += chunk
                            remaining -= len(chunk)
                        
                        request = BinaryProtocol.decode_request(data)
                        print(f"[Server {self.server_id}] [CN]: {request.get('command', 'unknown')}")
                    else:
                        # JSON[CN]
                        data = first_byte
                        while len(data) < length:
                            chunk = client_socket.recv(min(length - len(data), 4096))
                            if not chunk:
                                break
                            data += chunk
                        
                        request = json.loads(data.decode())
                        print(f"[Server {self.server_id}] [CN]JSON[CN]: {request.get('command', 'unknown')}")
                else:
                    # [CN]，[CN]
                    continue
                
                response = self._process_request(request)
                print(f"[Server {self.server_id}] [CN]process[CN]，[CN]: {response.get('status', 'unknown')}")
                
                # [CN]（[CN]）
                response_data = BinaryProtocol.encode_response(response)
                client_socket.sendall(response_data)
                
        except Exception as e:
            print(f"[Server {self.server_id}] process[CN]: {e}")
            import traceback
            traceback.print_exc()
        finally:
            client_socket.close()
    
    def _process_request(self, request: Dict) -> Dict:
        """process[CN]"""
        command = request.get('command')
        
        if command == 'query_node_vector':
            return self._handle_vector_node_query(request)
        elif command == 'get_status':
            return self._get_status()
        else:
            return {'status': 'error', 'message': f'[CN]: {command}'}
    
    def _multiprocess_vdpf_evaluation(self, serialized_key, num_nodes, num_batches):
        """[CN]VDPF[CN] - [CN]"""
        
        # create[CN]I/O
        shm_start = time.time()
        print(f"[Server {self.server_id}] create[CN]...")
        shm = shared_memory.SharedMemory(create=True, size=self.node_shares.nbytes)
        shared_array = np.ndarray(self.node_shares.shape, dtype=self.node_shares.dtype, buffer=shm.buf)
        shared_array[:] = self.node_shares[:]
        shm_time = time.time() - shm_start
        print(f"[Server {self.server_id}] [CN]create[CN]: {shm_time:.3f}[CN]")
        
        # [CN]allocate[CN]
        # calculate[CN]processnumber of nodes，[CN]
        nodes_per_process = num_nodes // self.vdpf_processes
        remaining_nodes = num_nodes % self.vdpf_processes
        
        # [CN]
        process_args = []
        current_node_start = 0
        
        for process_id in range(self.vdpf_processes):
            # calculate[CN]processnumber of nodes
            process_nodes = nodes_per_process + (1 if process_id < remaining_nodes else 0)
            
            if process_nodes == 0:
                continue
                
            # calculate[CN]
            node_start = current_node_start
            node_end = node_start + process_nodes
            
            # [CN]
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
                shm.name,  # [CN]
                self.node_shares.shape,  # [CN]
                self.node_shares.dtype,  # [CN]
                self.dataset  # [CN]Dataset[CN]
            )
            process_args.append(args)
            current_node_start = node_end
            
        # print[CN]allocate[CN]
        print(f"[Server {self.server_id}] [CN]allocate:")
        actual_node_counts = []
        for i, args in enumerate(process_args):
            process_id, start_batch, end_batch = args[0:3]
            batches = end_batch - start_batch
            # calculate[CN]
            actual_nodes = 0
            for batch_idx in range(start_batch, end_batch):
                batch_start = batch_idx * self.cache_batch_size
                batch_end = min(batch_start + self.cache_batch_size, num_nodes)
                actual_nodes += (batch_end - batch_start)
            actual_node_counts.append(actual_nodes)
            print(f"  - Process {process_id}: [CN] {start_batch}-{end_batch-1} ({batches} [CN], {actual_nodes} [CN])")
        
        # calculate[CN]
        if actual_node_counts:
            avg_nodes = np.mean(actual_node_counts)
            std_nodes = np.std(actual_node_counts)
            max_nodes = max(actual_node_counts)
            min_nodes = min(actual_node_counts)
            print(f"[Server {self.server_id}] [CN]: [CN]={avg_nodes:.0f}, [CN]={std_nodes:.1f}, [CN]={max_nodes}, [CN]={min_nodes}")
        
        # [CN]
        print(f"[Server {self.server_id}] [CN]VDPF[CN]（{len(process_args)}[CN]）")
        
        # [CN]create[CN]
        results = self.process_pool.map(evaluate_batch_range_process, process_args)
        
        # [CN]
        all_selector_shares = {}
        all_vector_shares = {}
        process_timings = []
        
        for process_result in results:
            all_selector_shares.update(process_result['selector_shares'])
            all_vector_shares.update(process_result['vector_shares'])
            if 'timing' in process_result:
                process_timings.append(process_result['timing'])
        
        # [CN]
        if process_timings:
            print(f"\n[Server {self.server_id}] [CN]1[CN]:")
            avg_vdpf_init = np.mean([t['vdpf_init'] for t in process_timings]) * 1000
            avg_deserialize = np.mean([t['deserialize'] for t in process_timings]) * 1000
            avg_shm_connect = np.mean([t['shm_connect'] for t in process_timings]) * 1000
            
            # [CN]
            max_vdpf_eval = max([t['vdpf_eval'] for t in process_timings]) * 1000
            max_total_time = max([t['total'] for t in process_timings]) * 1000
            
            # CPU[CN]（[CN]）
            total_cpu_time = sum([t['total'] for t in process_timings]) * 1000
            total_vdpf_cpu = sum([t['vdpf_eval'] for t in process_timings]) * 1000
            
            print(f"  === [CN] ===")
            print(f"  - [CN]: {max_total_time:.1f}ms ([CN])")
            print(f"  - [CN]VDPF[CN]: {max_vdpf_eval:.1f}ms")
            print(f"  - [CN]: {total_cpu_time/max_total_time/self.vdpf_processes*100:.1f}%")
            print(f"  ")
            print(f"  === [CN] ===")
            print(f"  - [CN]VDPFinstancecreate: {avg_vdpf_init:.1f}ms")
            print(f"  - [CN]: {avg_deserialize:.1f}ms")
            print(f"  - [CN]connect: {avg_shm_connect:.1f}ms")
            print(f"  - [CN]create: {shm_time*1000:.1f}ms")
            print(f"  ")
            print(f"  === CPU[CN] ===")
            print(f"  - [CN]CPU[CN]: {total_cpu_time:.1f}ms")
            print(f"  - [CN]VDPFcalculate[CN]: {total_vdpf_cpu:.1f}ms")
        
        print(f"[Server {self.server_id}] [CN]VDPF[CN]，[CN] {len(all_selector_shares)} [CN]")
        
        # [CN]
        shm.close()
        shm.unlink()
        
        return all_selector_shares, all_vector_shares
    
    def _save_exchange_data(self, query_id: str, e_shares: np.ndarray, f_shares: np.ndarray):
        """[CN]（[CN]）"""
        
        # [CN].npz[CN]，[CN]I/O[CN]
        filename = f"server_{self.server_id}_query_{query_id}_data.npz"
        filepath = os.path.join(self.exchange_dir, filename)
        
        # [CN]calculate（[CN]）
        data_hash = hashlib.md5((e_shares.tobytes() + f_shares.tobytes())).hexdigest()
        
        np.savez(filepath, 
                 e_shares=e_shares, 
                 f_shares=f_shares,
                 hash=data_hash)
        
        # print(f"[Server {self.server_id}] [CN] {filename}（[CN]）")
    
    def _load_other_servers_data(self, query_id: str, num_nodes: int) -> Tuple[Dict, Dict]:
        """[CN]"""
        
        all_e_from_others = {}
        all_f_from_others = {}
        
        for other_id in [1, 2, 3]:
            if other_id == self.server_id:
                continue
            
            filename = f"server_{other_id}_query_{query_id}_data.npz"
            filepath = os.path.join(self.exchange_dir, filename)
            
            # [CN]（[CN]）
            max_wait = 30
            for i in range(max_wait):
                if os.path.exists(filepath):
                    break
                time.sleep(1)
            
            if os.path.exists(filepath):
                # [CN].npz[CN]（[CN]，[CN]）
                data = np.load(filepath)
                all_e_from_others[other_id] = data['e_shares']
                all_f_from_others[other_id] = data['f_shares']
                # print(f"[Server {self.server_id}] [CN] Server {other_id} [CN]（[CN]）")
                
                # [CN]：[CN]
                if 'hash' in data:
                    loaded_hash = str(data['hash'])
                    expected_hash = hashlib.md5((data['e_shares'].tobytes() + data['f_shares'].tobytes())).hexdigest()
                    if loaded_hash != expected_hash:
                        pass  # print(f"[Server {self.server_id}] [CN]: Server {other_id} [CN]")
            else:
                pass  # print(f"[Server {self.server_id}] [CN]: [CN] Server {other_id} [CN]")
        
        return all_e_from_others, all_f_from_others
    
    def _handle_vector_node_query(self, request: Dict) -> Dict:
        """[CN] - [CN]"""
        try:
            serialized_key = request['dpf_key']
            query_id = request.get('query_id', 'unknown')
            
            print(f"[Server {self.server_id}] process[CN]（[CN]）")
            
            # 1. Deserialize keys（[CN]，[CN]）
            key = self.dpf_wrapper._deserialize_key(serialized_key)
            
            # 2. initialize
            start_time = time.time()
            num_nodes = len(self.node_shares)
            vector_dim = self.node_shares.shape[1]  # [CN]Vector dimension
            result_accumulator = np.zeros(vector_dim, dtype=np.int64)
            
            print(f"[Server {self.server_id}] [CN]: {num_nodes}, [CN]: {self.cache_batch_size}")
            # print(f"[Server {self.server_id}] DEBUG: Vector dimension: {vector_dim}, [CN]ID: {query_id}")
            
            # 3. [CN]1：[CN]VDPF[CN]（[CN])
            print(f"[Server {self.server_id}] [CN]1: [CN]VDPF[CN] ({self.vdpf_processes} [CN])...")
            phase1_start = time.time()
            
            num_batches = (num_nodes + self.cache_batch_size - 1) // self.cache_batch_size
            print(f"[Server {self.server_id}] [CN]process {num_batches} [CN]，[CN] {self.vdpf_processes} [CN]")
            
            # [CN]VDPF
            all_selector_shares, all_vector_shares = self._multiprocess_vdpf_evaluation(
                serialized_key, num_nodes, num_batches)
            
            phase1_time = time.time() - phase1_start
            total_ops_per_sec = num_nodes / phase1_time if phase1_time > 0 else 0
            print(f"[Server {self.server_id}] [CN]1[CN]（[CN]），[CN] {phase1_time:.2f}[CN], [CN] {total_ops_per_sec:.0f} ops/sec")
            
            # # DEBUG: [CN]VDPF[CN]
            # non_zero_count = sum(1 for v in all_selector_shares.values() if v != 0)
            # print(f"[Server {self.server_id}] DEBUG: VDPF[CN]: {non_zero_count}")
            # if non_zero_count > 0:
            #     for idx, val in list(all_selector_shares.items())[:5]:  # print[CN]5[CN]
            #         if val != 0:
            #             print(f"[Server {self.server_id}] DEBUG: [CN] {idx} [CN]VDPF[CN]: {val}")
            
            # 3.5. [CN]
            # print(f"[Server {self.server_id}] [CN]...")
            self._file_sync_barrier(query_id, "phase1")
            
            # 4. [CN]2：[CN]e/fcalculate（[CN]）
            # print(f"[Server {self.server_id}] [CN]2: [CN]e/fcalculate...")
            phase2_start = time.time()
            
            all_e_shares = np.zeros(num_nodes, dtype=np.uint64)
            all_f_shares = np.zeros((num_nodes, vector_dim), dtype=np.uint64)
            all_computation_states = {}
            
            # [CN]process，[CN]
            for batch_idx in range(num_batches):
                batch_start = batch_idx * self.cache_batch_size
                batch_end = min(batch_start + self.cache_batch_size, num_nodes)
                batch_size = batch_end - batch_start
                
                # print(f"[Server {self.server_id}] e/fcalculate[CN] {batch_idx+1}/{num_batches}")
                
                # [CN]（[CN]）
                batch_triples = []
                for _ in range(batch_size):
                    batch_triples.append(self.mult_server.get_next_triple())
                
                # [CN]process[CN]
                for local_idx in range(batch_size):
                    global_idx = batch_start + local_idx
                    computation_id = f"query_{query_id}_pos{global_idx}"
                    
                    a, b, c = batch_triples[local_idx]
                    
                    e_share = (all_selector_shares[global_idx] - a) % self.field_size
                    all_e_shares[global_idx] = e_share
                    
                    # [CN]calculate[CN]f[CN]
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
            # print(f"[Server {self.server_id}] [CN]2[CN]，[CN] {phase2_time:.2f}[CN]")
            
            # 5. [CN]3：[CN]（[CN]）
            # print(f"[Server {self.server_id}] [CN]3: [CN]...")
            phase3_start = time.time()
            
            # [CN]
            self._save_exchange_data(query_id, all_e_shares, all_f_shares)
            
            # [CN]
            self._file_sync_barrier(query_id, "phase3_save")
            
            # [CN]
            all_e_from_others, all_f_from_others = self._load_other_servers_data(query_id, num_nodes)
            
            phase3_time = time.time() - phase3_start
            # print(f"[Server {self.server_id}] [CN]3[CN]（[CN]），[CN] {phase3_time:.2f}[CN]")
            
            # 6. [CN]4：[CN]calculate（[CN]NumPy[CN]）
            # print(f"[Server {self.server_id}] [CN]4: [CN]calculate...")
            phase4_start = time.time()
            
            # [CN]calculate[CN]，[CN]
            lagrange_1 = 2
            lagrange_2 = self.field_size - 1
            
            # [CN]process[CN]calculate
            for batch_idx in range(num_batches):
                batch_start = batch_idx * self.cache_batch_size
                batch_end = min(batch_start + self.cache_batch_size, num_nodes)
                batch_size = batch_end - batch_start
                
                # print(f"[Server {self.server_id}] NumPy[CN] {batch_idx+1}/{num_batches} ({batch_size} [CN])")
                
                # 1. [CN] - [CN]
                batch_e_shares_local = all_e_shares[batch_start:batch_end]  # shape: (batch_size,)
                batch_f_shares_local = all_f_shares[batch_start:batch_end, :]  # shape: (batch_size, vector_dim)
                
                # 2. [CN]servers[CN]
                # e_shares_matrix: shape (batch_size, 3)
                e_shares_matrix = np.zeros((batch_size, 3), dtype=np.uint64)
                e_shares_matrix[:, self.server_id - 1] = batch_e_shares_local
                
                # f_shares_matrix: shape (batch_size, vector_dim, 3)
                f_shares_matrix = np.zeros((batch_size, vector_dim, 3), dtype=np.uint64)
                f_shares_matrix[:, :, self.server_id - 1] = batch_f_shares_local
                
                # 3. [CN]
                for other_id, other_e_shares in all_e_from_others.items():
                    e_shares_matrix[:, other_id - 1] = other_e_shares[batch_start:batch_end]
                
                for other_id, other_f_shares in all_f_from_others.items():
                    f_shares_matrix[:, :, other_id - 1] = other_f_shares[batch_start:batch_end, :]
                
                # 4. [CN]e[CN] - [CN]process[CN]
                # [CN]: e = e1 * 2 + e2 * (-1)
                batch_e_reconstructed = (e_shares_matrix[:, 0] * lagrange_1 + 
                                       e_shares_matrix[:, 1] * lagrange_2) % self.field_size
                # shape: (batch_size,)
                
                # 5. [CN]f[CN] - [CN]process[CN]
                # f_reconstructed: shape (batch_size, vector_dim)
                batch_f_reconstructed = (f_shares_matrix[:, :, 0] * lagrange_1 + 
                                       f_shares_matrix[:, :, 1] * lagrange_2) % self.field_size
                
                # 6. [CN]
                batch_triples = []
                for local_idx in range(batch_size):
                    global_idx = batch_start + local_idx
                    state = all_computation_states[global_idx]
                    a, b, c = state['triple']
                    batch_triples.append((a, b, c))
                
                # [CN]NumPy[CN]
                batch_a = np.array([t[0] for t in batch_triples], dtype=np.uint64)  # shape: (batch_size,)
                batch_b = np.array([t[1] for t in batch_triples], dtype=np.uint64)  # shape: (batch_size,)
                batch_c = np.array([t[2] for t in batch_triples], dtype=np.uint64)  # shape: (batch_size,)
                
                # 7. [CN]calculate[CN] - [CN]process[CN]
                # [CN]calculate batch_size × vector_dim [CN]
                
                # [CN]: (batch_size, 1) [CN] (batch_size, vector_dim)
                batch_e_expanded = batch_e_reconstructed[:, np.newaxis]  # shape: (batch_size, 1)
                batch_a_expanded = batch_a[:, np.newaxis]  # shape: (batch_size, 1)
                batch_b_expanded = batch_b[:, np.newaxis]  # shape: (batch_size, 1)
                batch_c_expanded = batch_c[:, np.newaxis]  # shape: (batch_size, 1)
                
                # calculate[CN]: result = c + e*b + f*a + e*f (mod field_size)
                batch_result = batch_c_expanded  # shape: (batch_size, vector_dim)
                batch_result = (batch_result + batch_e_expanded * batch_b_expanded) % self.field_size
                batch_result = (batch_result + batch_f_reconstructed * batch_a_expanded) % self.field_size
                batch_result = (batch_result + batch_e_expanded * batch_f_reconstructed) % self.field_size
                
                # 8. [CN] - [CN]Vector dimension[CN]
                batch_contribution = np.sum(batch_result, axis=0) % self.field_size  # shape: (vector_dim,)
                result_accumulator = (result_accumulator + batch_contribution) % self.field_size
                
                # # DEBUG: [CN]
                # if batch_idx == 0:
                #     print(f"[Server {self.server_id}] DEBUG: [CN]1[CN]5[CN]: {batch_contribution[:5]}")
                #     print(f"[Server {self.server_id}] DEBUG: [CN]5[CN]: {result_accumulator[:5]}")
                
                # 9. [CN]calculate[CN]
                for local_idx in range(batch_size):
                    global_idx = batch_start + local_idx
                    state = all_computation_states[global_idx]
                    computation_id = state['computation_id']
                    if computation_id in self.mult_server.computation_cache:
                        del self.mult_server.computation_cache[computation_id]
                
                # print(f"[Server {self.server_id}] [CN] {batch_idx+1} [CN]")
            
            phase4_time = time.time() - phase4_start
            total_time = time.time() - start_time
            
            print(f"[Server {self.server_id}] [CN]（[CN]）:")
            print(f"  [CN]1 ([CN]VDPF): {phase1_time:.2f}[CN]")
            print(f"  [CN]2 ([CN]e/fcalculate): {phase2_time:.2f}[CN]")
            print(f"  [CN]3 ([CN]): {phase3_time:.2f}[CN]") 
            print(f"  [CN]4 ([CN]): {phase4_time:.2f}[CN]")
            print(f"  [CN]: {total_time:.2f}[CN]")
            
            # [CN]，[CN]
            self._file_sync_barrier(query_id, "phase4_complete")
            
            # [CN]，[CN]
            # [CN]，[CN]
            self._cleanup_data_files_only(query_id)
            
            # return[CN]
            # print(f"[Server {self.server_id}] [CN]...")
            
            # [CN]
            try:
                result_list = [int(x) % (2**32) for x in result_accumulator]
                # print(f"[Server {self.server_id}] [CN]，[CN]: {len(result_list)}")
                # print(f"[Server {self.server_id}] DEBUG: [CN]5[CN]: {result_list[:5]}")
                # print(f"[Server {self.server_id}] DEBUG: [CN]: [{min(result_list)}, {max(result_list)}]")
            except Exception as e:
                print(f"[Server {self.server_id}] [CN]: {e}")
                result_list = [0] * vector_dim  # [CN]
            
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
            
            # print(f"[Server {self.server_id}] [CN]")
            return response
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}
    
    def _file_sync_barrier(self, query_id: str, phase: str):
        """[CN]"""
        # create[CN]
        marker_file = f"server_{self.server_id}_query_{query_id}_{phase}_ready"
        marker_path = os.path.join(self.exchange_dir, marker_file)
        
        with open(marker_path, 'w') as f:
            f.write(str(time.time()))
        
        # [CN]
        for other_id in [1, 2, 3]:
            if other_id == self.server_id:
                continue
            
            other_marker = f"server_{other_id}_query_{query_id}_{phase}_ready"
            other_path = os.path.join(self.exchange_dir, other_marker)
            
            max_wait = 60
            for i in range(max_wait):
                if os.path.exists(other_path):
                    break
                time.sleep(0.1)  # [CN]
            
            if not os.path.exists(other_path):
                print(f"[Server {self.server_id}] [CN]: Server {other_id} [CN] {phase}")
    
    def _cleanup_query_files(self, query_id: str):
        """[CN]"""
        for filename in os.listdir(self.exchange_dir):
            if query_id in filename:
                # [CN]
                try:
                    os.remove(os.path.join(self.exchange_dir, filename))
                except:
                    pass
    
    def _cleanup_data_files_only(self, query_id: str):
        """[CN]，[CN]"""
        for filename in os.listdir(self.exchange_dir):
            if query_id in filename and filename.endswith('_data.npz'):
                # [CN]
                try:
                    os.remove(os.path.join(self.exchange_dir, filename))
                except:
                    pass
    
    def _get_status(self) -> Dict:
        """[CN]"""
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
    """[CN]"""
    # [CN]start[CN]'spawn'，[CN]initialize
    multiprocessing.set_start_method('spawn', force=True)
    
    # [CN]
    parser = argparse.ArgumentParser(description='[CN]')
    parser.add_argument('--server-id', type=int, required=True, choices=[1, 2, 3],
                        help='[CN]ID (1, 2, [CN] 3)')
    parser.add_argument('--dataset', type=str, default='laion', choices=['laion', 'siftsmall', 'tripclick', 'ms_marco', 'nfcorpus'],
                        help='Dataset[CN] ([CN]: laion)')
    parser.add_argument('--vdpf-processes', type=int, default=4,
                        help='VDPF[CN] ([CN]: 4, [CN]: 1-16)')
    
    args = parser.parse_args()
    
    if args.vdpf_processes < 1 or args.vdpf_processes > 16:
        print("[CN]: vdpf_processes [CN] 1-16 [CN]")
        sys.exit(1)
    
    print(f"start[CN] {args.server_id}，Dataset: {args.dataset}，[CN] {args.vdpf_processes} [CN]VDPF[CN]")
    server = MemoryMappedOptimizedServer(args.server_id, args.dataset, args.vdpf_processes)
    server.start()


if __name__ == "__main__":
    main()