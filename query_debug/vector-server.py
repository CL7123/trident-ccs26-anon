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
import psutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('~/trident/src')

from dpf_wrapper import VDPFVectorWrapper
from secure_multiplication import NumpyMultiplicationServer
from basic_functionalities import get_config, Share, MPC23SSS


# Global function for process pool (must be defined at module top level)
def evaluate_batch_range_process(args):
    """Process pool worker: Evaluate specified batch range of VDPF"""
    process_id, start_batch, end_batch, cache_batch_size, num_nodes, serialized_key, server_id, node_shares_path, dataset_name = args
    
    print(f"[Process {process_id}] Start evaluating batch {start_batch}-{end_batch-1}")
    
    # Each process creates independentVDPFinstance
    from dpf_wrapper import VDPFVectorWrapper
    dpf_wrapper = VDPFVectorWrapper(dataset_name=dataset_name)
    
    # Deserialize keys
    key = dpf_wrapper._deserialize_key(serialized_key)
    
    # [CN]（[CN]）
    node_shares = np.load(node_shares_path)
    
    local_selector_shares = {}
    local_vector_shares = {}
    process_start_time = time.time()
    
    for batch_idx in range(start_batch, end_batch):
        batch_start = batch_idx * cache_batch_size
        batch_end = min(batch_start + cache_batch_size, num_nodes)
        batch_size = batch_end - batch_start
        
        # [CN]
        batch_data = node_shares[batch_start:batch_end].copy()
        
        # [CN]process[CN]
        for local_idx in range(batch_size):
            global_idx = batch_start + local_idx
            
            # VDPF[CN]
            selector_share = dpf_wrapper.eval_at_position(key, global_idx, server_id)
            local_selector_shares[global_idx] = selector_share
            
            # [CN]
            local_vector_shares[global_idx] = batch_data[local_idx]
    
    process_time = time.time() - process_start_time
    evaluated_nodes = sum(min(cache_batch_size, num_nodes - b * cache_batch_size) 
                        for b in range(start_batch, end_batch))
    ops_per_sec = evaluated_nodes / process_time if process_time > 0 else 0
    
    print(f"[Process {process_id}] [CN]: {process_time:.3f}[CN], "
          f"{evaluated_nodes} [CN], {ops_per_sec:.0f} ops/sec")
    
    return {
        'selector_shares': local_selector_shares,
        'vector_shares': local_vector_shares
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
        self.process = psutil.Process()
        self.memory_records = []
        self._record_memory("[CN]initialize[CN]")
        
        # [CN]
        self.host = f"192.168.50.2{server_id}"
        self.port = 8000 + server_id
        
        # initialize[CN]
        self.dpf_wrapper = VDPFVectorWrapper(dataset_name=dataset)
        self.mult_server = NumpyMultiplicationServer(server_id, self.config)
        self._record_memory("[CN]initialize[CN]")
        
        # [CN]
        self._load_data()
        
        # [CN]（[CN]）
        self.exchange_dir = "/tmp/mpc_exchange"
        os.makedirs(self.exchange_dir, exist_ok=True)
        
        # ===== [CN] =====
        self.cache_batch_size = 2000  # [CN]
        self.vdpf_processes = vdpf_processes  # [CN]VDPF[CN]
        self.worker_threads = 4  # [CN]
        
        # [CN]create[CN]
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.worker_threads,
            thread_name_prefix=f"Server{server_id}-General"
        )
        
        print(f"[Server {self.server_id}] [CN]")
        print(f"[Server {self.server_id}] [CN]: {self.cache_batch_size}")
        print(f"[Server {self.server_id}] VDPF[CN]: {self.vdpf_processes}")
        print(f"[Server {self.server_id}] [CN]: {self.exchange_dir}")
    
    def _record_memory(self, stage_name: str):
        """[CN]"""
        try:
            memory_info = self.process.memory_info()
            memory_record = {
                'stage': stage_name,
                'timestamp': time.time(),
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
                'percent': self.process.memory_percent()
            }
            self.memory_records.append(memory_record)
            print(f"[Server {self.server_id} [CN]] {stage_name}: {memory_record['rss_mb']:.2f} MB")
        except Exception as e:
            print(f"Server {self.server_id} [CN]: {e}")
    
    def _get_memory_summary(self):
        """[CN]"""
        if not self.memory_records:
            return {}
        
        peak_rss = max(r['rss_mb'] for r in self.memory_records)
        baseline_rss = self.memory_records[0]['rss_mb']
        
        return {
            'server_id': self.server_id,
            'baseline_mb': baseline_rss,
            'peak_mb': peak_rss,
            'increase_mb': peak_rss - baseline_rss,
            'records': self.memory_records
        }
        
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
        self._record_memory("[CN]")
        
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
                print(f"[Server {self.server_id}] [CN]: {request.get('command', 'unknown')}")
                
                response = self._process_request(request)
                print(f"[Server {self.server_id}] [CN]process[CN]，[CN]: {response.get('status', 'unknown')}")
                
                response_data = json.dumps(response).encode()
                client_socket.sendall(len(response_data).to_bytes(4, 'big'))
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
        """[CN]VDPF[CN] - [CN]（[CN]）"""
        
        # [CN]allocate[CN]
        batches_per_process = max(1, num_batches // self.vdpf_processes)
        remaining_batches = num_batches % self.vdpf_processes
        
        # [CN]
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
                    self.nodes_path,  # [CN]
                    self.dataset  # [CN]Dataset[CN]
                )
                process_args.append(args)
                current_batch = end_batch
        
        # [CN]
        print(f"[Server {self.server_id}] start {len(process_args)} [CN]VDPF[CN]")
        
        with Pool(processes=self.vdpf_processes) as pool:
            results = pool.map(evaluate_batch_range_process, process_args)
        
        # [CN]
        all_selector_shares = {}
        all_vector_shares = {}
        
        for process_result in results:
            all_selector_shares.update(process_result['selector_shares'])
            all_vector_shares.update(process_result['vector_shares'])
        
        print(f"[Server {self.server_id}] [CN]VDPF[CN]，[CN] {len(all_selector_shares)} [CN]")
        
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
        
        print(f"[Server {self.server_id}] [CN] {filename}（[CN]）")
    
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
                print(f"[Server {self.server_id}] [CN] Server {other_id} [CN]（[CN]）")
                
                # [CN]：[CN]
                if 'hash' in data:
                    loaded_hash = str(data['hash'])
                    expected_hash = hashlib.md5((data['e_shares'].tobytes() + data['f_shares'].tobytes())).hexdigest()
                    if loaded_hash != expected_hash:
                        print(f"[Server {self.server_id}] [CN]: Server {other_id} [CN]")
            else:
                print(f"[Server {self.server_id}] [CN]: [CN] Server {other_id} [CN]")
        
        return all_e_from_others, all_f_from_others
    
    def _handle_vector_node_query(self, request: Dict) -> Dict:
        """[CN] - [CN]"""
        try:
            self._record_memory("[CN]process[CN]")
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
            print(f"[Server {self.server_id}] DEBUG: Vector dimension: {vector_dim}, [CN]ID: {query_id}")
            
            # 3. [CN]1：[CN]VDPF[CN]（[CN])
            print(f"[Server {self.server_id}] [CN]1: [CN]VDPF[CN] ({self.vdpf_processes} [CN])...")
            phase1_start = time.time()
            
            num_batches = (num_nodes + self.cache_batch_size - 1) // self.cache_batch_size
            print(f"[Server {self.server_id}] [CN]process {num_batches} [CN]，[CN] {self.vdpf_processes} [CN]")
            
            # [CN]VDPF
            all_selector_shares, all_vector_shares = self._multiprocess_vdpf_evaluation(
                serialized_key, num_nodes, num_batches)
            
            self._record_memory("[CN]1 VDPF[CN]")
            phase1_time = time.time() - phase1_start
            total_ops_per_sec = num_nodes / phase1_time if phase1_time > 0 else 0
            print(f"[Server {self.server_id}] [CN]1[CN]（[CN]），[CN] {phase1_time:.2f}[CN], [CN] {total_ops_per_sec:.0f} ops/sec")
            
            # DEBUG: [CN]VDPF[CN]
            non_zero_count = sum(1 for v in all_selector_shares.values() if v != 0)
            print(f"[Server {self.server_id}] DEBUG: VDPF[CN]: {non_zero_count}")
            if non_zero_count > 0:
                for idx, val in list(all_selector_shares.items())[:5]:  # print[CN]5[CN]
                    if val != 0:
                        print(f"[Server {self.server_id}] DEBUG: [CN] {idx} [CN]VDPF[CN]: {val}")
            
            # 3.5. [CN]
            print(f"[Server {self.server_id}] [CN]...")
            self._file_sync_barrier(query_id, "phase1")
            
            # 4. [CN]2：[CN]e/fcalculate（[CN]）
            print(f"[Server {self.server_id}] [CN]2: [CN]e/fcalculate...")
            phase2_start = time.time()
            
            all_e_shares = np.zeros(num_nodes, dtype=np.uint64)
            all_f_shares = np.zeros((num_nodes, vector_dim), dtype=np.uint64)
            all_computation_states = {}
            
            # [CN]process，[CN]
            for batch_idx in range(num_batches):
                batch_start = batch_idx * self.cache_batch_size
                batch_end = min(batch_start + self.cache_batch_size, num_nodes)
                batch_size = batch_end - batch_start
                
                print(f"[Server {self.server_id}] e/fcalculate[CN] {batch_idx+1}/{num_batches}")
                
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
            print(f"[Server {self.server_id}] [CN]2[CN]，[CN] {phase2_time:.2f}[CN]")
            self._record_memory("[CN]2 e/fcalculate[CN]")
            
            # 5. [CN]3：[CN]（[CN]）
            print(f"[Server {self.server_id}] [CN]3: [CN]...")
            phase3_start = time.time()
            
            # [CN]
            self._save_exchange_data(query_id, all_e_shares, all_f_shares)
            
            # [CN]
            self._file_sync_barrier(query_id, "phase3_save")
            
            # [CN]
            all_e_from_others, all_f_from_others = self._load_other_servers_data(query_id, num_nodes)
            
            phase3_time = time.time() - phase3_start
            print(f"[Server {self.server_id}] [CN]3[CN]（[CN]），[CN] {phase3_time:.2f}[CN]")
            self._record_memory("[CN]3 [CN]")
            
            # 6. [CN]4：[CN]calculate（[CN]NumPy[CN]）
            print(f"[Server {self.server_id}] [CN]4: [CN]calculate...")
            phase4_start = time.time()
            
            # [CN]calculate[CN]，[CN]
            lagrange_1 = 2
            lagrange_2 = self.field_size - 1
            
            # [CN]process[CN]calculate
            for batch_idx in range(num_batches):
                batch_start = batch_idx * self.cache_batch_size
                batch_end = min(batch_start + self.cache_batch_size, num_nodes)
                batch_size = batch_end - batch_start
                
                print(f"[Server {self.server_id}] NumPy[CN] {batch_idx+1}/{num_batches} ({batch_size} [CN])")
                
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
                
                # DEBUG: [CN]
                if batch_idx == 0:
                    print(f"[Server {self.server_id}] DEBUG: [CN]1[CN]5[CN]: {batch_contribution[:5]}")
                    print(f"[Server {self.server_id}] DEBUG: [CN]5[CN]: {result_accumulator[:5]}")
                
                # 9. [CN]calculate[CN]
                for local_idx in range(batch_size):
                    global_idx = batch_start + local_idx
                    state = all_computation_states[global_idx]
                    computation_id = state['computation_id']
                    if computation_id in self.mult_server.computation_cache:
                        del self.mult_server.computation_cache[computation_id]
                
                print(f"[Server {self.server_id}] [CN] {batch_idx+1} [CN]")
            
            phase4_time = time.time() - phase4_start
            total_time = time.time() - start_time
            self._record_memory("[CN]4 [CN]calculate[CN]")
            
            print(f"[Server {self.server_id}] [CN]（[CN]）:")
            print(f"  [CN]1 ([CN]VDPF): {phase1_time:.2f}[CN]")
            print(f"  [CN]2 ([CN]e/fcalculate): {phase2_time:.2f}[CN]")
            print(f"  [CN]3 ([CN]): {phase3_time:.2f}[CN]") 
            print(f"  [CN]4 ([CN]): {phase4_time:.2f}[CN]")
            print(f"  [CN]: {total_time:.2f}[CN]")
            
            # [CN]，[CN]
            self._file_sync_barrier(query_id, "phase4_complete")
            
            # [CN]
            self._cleanup_query_files(query_id)
            
            # return[CN]
            print(f"[Server {self.server_id}] [CN]...")
            
            # [CN]
            memory_summary = self._get_memory_summary()
            if memory_summary:
                print(f"[Server {self.server_id}] === [CN] ===")
                print(f"  [CN]: {memory_summary['baseline_mb']:.2f} MB")
                print(f"  [CN]: {memory_summary['peak_mb']:.2f} MB")
                print(f"  [CN]: {memory_summary['increase_mb']:.2f} MB")
            
            # [CN]
            try:
                result_list = [int(x) % (2**32) for x in result_accumulator]
                print(f"[Server {self.server_id}] [CN]，[CN]: {len(result_list)}")
                print(f"[Server {self.server_id}] DEBUG: [CN]5[CN]: {result_list[:5]}")
                print(f"[Server {self.server_id}] DEBUG: [CN]: [{min(result_list)}, {max(result_list)}]")
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
            
            print(f"[Server {self.server_id}] [CN]")
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
            if query_id in filename and filename.endswith('_data.npz'):
                # [CN]，[CN]
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