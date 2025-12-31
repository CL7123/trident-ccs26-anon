#!/usr/bin/env python3

import sys
import os
import socket
import json
import numpy as np
import time
import concurrent.futures
import random
import argparse
import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('~/trident/src')
sys.path.append('~/trident/query-opti')  # Add optimization directory

from dpf_wrapper import VDPFVectorWrapper
from dpf_wrapper_optimized import OptimizedVDPFVectorWrapper  # Import optimized wrapper
from binary_protocol import BinaryProtocol  # Import binary protocol
from basic_functionalities import get_config, MPC23SSS, Share
from share_data import DatasetLoader


class MultiprocessConfigurableTestClient:
    """[CN]"""
    
    def __init__(self, dataset: str = "laion"):
        self.dataset = dataset
        self.config = get_config(dataset)
        self.dpf_wrapper = OptimizedVDPFVectorWrapper(dataset_name=dataset)
        self.mpc = MPC23SSS(self.config)
        
        # [CN]
        data_dir = f"~/trident/dataset/{dataset}"
        loader = DatasetLoader(data_dir)
        self.original_nodes = loader.load_nodes()
        print(f"[CN] {len(self.original_nodes)} [CN]")
        
        # [CN]
        self.servers = [
            ("192.168.50.21", 8001),
            ("192.168.50.22", 8002),
            ("192.168.50.23", 8003)
        ]
        
        self.connections = {}
        
    def connect_to_servers(self):
        """connect[CN]"""
        for i, (host, port) in enumerate(self.servers):
            server_id = i + 1
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((host, port))
                self.connections[server_id] = sock
            except Exception:
                pass
        
    def _send_request(self, server_id: int, request: dict) -> dict:
        """[CN]send[CN]"""
        sock = self.connections[server_id]
        
        # [CN]send[CN]（[CN]）
        if 'dpf_key' in request:
            BinaryProtocol.send_binary_request(
                sock, 
                request['command'],
                request['dpf_key'],  # [CN]bytes
                request.get('query_id')
            )
            # receive[CN]
            return BinaryProtocol.receive_response(sock)
        else:
            # [CN]JSON
            request_data = json.dumps(request).encode()
            sock.sendall(len(request_data).to_bytes(4, 'big'))
            sock.sendall(request_data)
            
            length_bytes = sock.recv(4)
            length = int.from_bytes(length_bytes, 'big')
            data = b''
            while len(data) < length:
                chunk = sock.recv(min(length - len(data), 4096))
                data += chunk
        
        return json.loads(data.decode())
    
    def test_mmap_query(self, node_id: int = 1723):
        """[CN]"""
        # [CN]VDPF[CN]
        keys = self.dpf_wrapper.generate_keys('node', node_id)
        
        # [CN]query_id（[CN]，[CN]ID）
        query_id = f'multiprocess_test_{time.time()}_{node_id}'
        
        # [CN]
        start_time = time.time()
        
        def query_server(server_id):
            request = {
                'command': 'query_node_vector',
                'dpf_key': keys[server_id - 1],
                'query_id': query_id  # [CN]query_id
            }
            response = self._send_request(server_id, request)
            return server_id, response
        
        # [CN]
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(query_server, sid) for sid in self.connections]
            results = {}
            
            for future in concurrent.futures.as_completed(futures):
                server_id, response = future.result()
                results[server_id] = response
        
        if all(r.get('status') == 'success' for r in results.values()):
            # [CN]
            timings = {}
            for server_id, result in results.items():
                timing = result.get('timing', {})
                timings[server_id] = {
                    'phase1': timing.get('phase1_time', 0) / 1000,  # [CN]
                    'phase2': timing.get('phase2_time', 0) / 1000,
                    'phase3': timing.get('phase3_time', 0) / 1000,
                    'phase4': timing.get('phase4_time', 0) / 1000,
                    'total': timing.get('total', 0) / 1000
                }
            
            # calculate[CN]
            avg_timings = {}
            for phase in ['phase1', 'phase2', 'phase3', 'phase4', 'total']:
                avg_timings[phase] = np.mean([t[phase] for t in timings.values()])
            
            # [CN]
            final_result = self._reconstruct_final_result(results)
            
            # # DEBUG: print[CN]
            # print(f"\nDEBUG: [CN]:")
            # for sid, result in results.items():
            #     if result.get('status') == 'success':
            #         shares = result['result_share']
            #         print(f"  Server {sid} [CN]5[CN]: {shares[:5]}")
            
            # [CN]
            similarity = self._verify_result(node_id, final_result)
            
            # [CN]print[CN]
            print(f"\n[CN]:")
            print(f"  [CN]1 ([CN]VDPF[CN]): {avg_timings['phase1']:.2f}[CN]")
            print(f"  [CN]2 (e/fcalculate): {avg_timings['phase2']:.2f}[CN]")
            print(f"  [CN]3 ([CN]): {avg_timings['phase3']:.2f}[CN]")
            print(f"  [CN]4 ([CN]): {avg_timings['phase4']:.2f}[CN]")
            print(f"  [CN]: {avg_timings['total']:.2f}[CN]")
            if similarity is not None:
                print(f"  [CN]: {similarity:.6f}")
            
            return avg_timings, final_result
            
        else:
            print("❌ [CN]:")
            for server_id, result in results.items():
                if result.get('status') != 'success':
                    print(f"  MultiProcess Server {server_id}: {result.get('message', 'Unknown error')}")
            return None, None
    
    def _reconstruct_final_result(self, results):
        """[CN]"""
        # [CN]servers[CN]
        server_ids = []
        for server_id in [1, 2, 3]:
            if server_id in results and results[server_id].get('status') == 'success':
                server_ids.append(server_id)
        
        if len(server_ids) < 2:
            print("[CN]：[CN]2servers[CN]")
            # [CN]Dataset[CN]
            default_dim = 512 if self.dataset == "laion" else 128
            return np.zeros(default_dim, dtype=np.float32)
        
        # [CN]
        server_ids = sorted(server_ids)[:2]
        
        # [CN]Vector dimension
        first_result = results[server_ids[0]]['result_share']
        vector_dim = len(first_result)
        
        # [CN]
        reconstructed_vector = np.zeros(vector_dim, dtype=np.float32)
        
        # [CN]
        if self.dataset == "siftsmall":
            scale_factor = 1048576  # 2^20 for siftsmall
        else:
            scale_factor = 536870912  # 2^29 for other datasets
        
        for i in range(vector_dim):
            shares = [
                Share(results[server_ids[0]]['result_share'][i], server_ids[0]),
                Share(results[server_ids[1]]['result_share'][i], server_ids[1])
            ]
            
            reconstructed = self.mpc.reconstruct(shares)
            
            # [CN]
            if reconstructed > self.config.prime // 2:
                signed = reconstructed - self.config.prime
            else:
                signed = reconstructed
            
            reconstructed_vector[i] = signed / scale_factor
        
        # # DEBUG: print[CN]
        # print(f"\nDEBUG: [CN]:")
        # print(f"  [CN]5[CN]: {reconstructed_vector[:5]}")
        # print(f"  [CN]: [{np.min(reconstructed_vector):.6f}, {np.max(reconstructed_vector):.6f}]")
        # print(f"  [CN]: {np.mean(reconstructed_vector):.6f}, [CN]: {np.std(reconstructed_vector):.6f}")
        
        return reconstructed_vector
    
    def _verify_result(self, node_id: int, reconstructed_vector: np.ndarray):
        """[CN]，return[CN]"""
        try:
            if node_id < len(self.original_nodes):
                original_vector = self.original_nodes[node_id]
                
                # # DEBUG: print[CN]
                # print(f"\nDEBUG: [CN] ([CN] {node_id}):")
                # print(f"  [CN]5[CN]: {original_vector[:5]}")
                # print(f"  [CN]: [{np.min(original_vector):.6f}, {np.max(original_vector):.6f}]")
                # print(f"  [CN]: {np.mean(original_vector):.6f}, [CN]: {np.std(original_vector):.6f}")
                
                # calculate[CN]
                dot_product = np.dot(reconstructed_vector, original_vector)
                norm_reconstructed = np.linalg.norm(reconstructed_vector)
                norm_original = np.linalg.norm(original_vector)
                
                if norm_reconstructed > 0 and norm_original > 0:
                    similarity = dot_product / (norm_reconstructed * norm_original)
                    return similarity
                else:
                    return None
            else:
                return None
                
        except Exception:
            return None
    
    def disconnect_from_servers(self):
        """[CN]connect"""
        for sock in self.connections.values():
            sock.close()
        self.connections.clear()


def generate_markdown_report(dataset, query_details, avg_phases, avg_similarity):
    """[CN]Markdown[CN]"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    markdown = f"""# Test results[CN] - {dataset}

**[CN]**: {timestamp}  
**Dataset**: {dataset}  
**[CN]**: {len(query_details)}

## [CN]

| [CN] | [CN]ID | [CN]1 (VDPF) | [CN]2 (e/f) | [CN]3 ([CN]) | [CN]4 ([CN]) | [CN] | [CN] |
|---------|--------|--------------|-------------|--------------|--------------|--------|-----------|
"""
    
    for q in query_details:
        markdown += f"| {q['query_num']} | {q['node_id']} | "
        markdown += f"{q['timings']['phase1']:.2f}s | "
        markdown += f"{q['timings']['phase2']:.2f}s | "
        markdown += f"{q['timings']['phase3']:.2f}s | "
        markdown += f"{q['timings']['phase4']:.2f}s | "
        markdown += f"{q['timings']['total']:.2f}s | "
        markdown += f"{q['similarity']:.6f} |\n"
    
    markdown += f"""
## [CN]

- **[CN]1 ([CN]VDPF[CN])**: {avg_phases['phase1']:.2f}[CN]
- **[CN]2 (e/fcalculate)**: {avg_phases['phase2']:.2f}[CN]
- **[CN]3 ([CN])**: {avg_phases['phase3']:.2f}[CN]
- **[CN]4 ([CN])**: {avg_phases['phase4']:.2f}[CN]
- **[CN]**: {avg_phases['total']:.2f}[CN]
- **[CN]**: {avg_similarity:.6f}

## [CN]

### [CN]
"""
    
    # calculate[CN]
    total_avg = avg_phases['total']
    phase1_pct = (avg_phases['phase1'] / total_avg) * 100
    phase2_pct = (avg_phases['phase2'] / total_avg) * 100
    phase3_pct = (avg_phases['phase3'] / total_avg) * 100
    phase4_pct = (avg_phases['phase4'] / total_avg) * 100
    
    markdown += f"""
- [CN]1 (VDPF[CN]): {phase1_pct:.1f}%
- [CN]2 (e/fcalculate): {phase2_pct:.1f}%
- [CN]3 ([CN]): {phase3_pct:.1f}%
- [CN]4 ([CN]): {phase4_pct:.1f}%

### [CN]
- [CN]: {total_avg:.2f}[CN]
- [CN]: {1/total_avg:.2f} [CN]/[CN]
"""
    
    return markdown


def main():
    """[CN]"""
    # [CN]
    parser = argparse.ArgumentParser(description='[CN]')
    parser.add_argument('--dataset', type=str, default='siftsmall', 
                        choices=['siftsmall', 'laion', 'tripclick', 'ms_marco', 'nfcorpus'],
                        help='Dataset[CN] ([CN]: siftsmall)')
    parser.add_argument('--num-queries', type=int, default=10,
                        help='[CN]Number of queries[CN] ([CN]: 10)')
    parser.add_argument('--no-report', action='store_true',
                        help='[CN]')
    
    args = parser.parse_args()
    
    print(f"=== [CN] - Dataset: {args.dataset} ===")
    
    client = MultiprocessConfigurableTestClient(args.dataset)
    
    try:
        client.connect_to_servers()
        
        if len(client.connections) == 0:
            print("[CN]：[CN]connect[CN]")
            return
        
        all_timings = []
        all_similarities = []
        query_details = []  # [CN]
        
        # [CN]
        total_nodes = len(client.original_nodes)
        
        # [CN]
        random_nodes = random.sample(range(total_nodes), min(args.num_queries, total_nodes))
        
        print(f"[CN] {len(random_nodes)} [CN]...\n")
        
        for i, node_id in enumerate(random_nodes):
            print(f"[CN] {i+1}/{len(random_nodes)}: [CN] {node_id}")
            timings, final_result = client.test_mmap_query(node_id=node_id)
            
            if timings:
                all_timings.append(timings)
                # [CN]
                similarity = client._verify_result(node_id, final_result)
                if similarity is not None:
                    all_similarities.append(similarity)
                
                # [CN]
                query_details.append({
                    'query_num': i + 1,
                    'node_id': node_id,
                    'timings': timings,
                    'similarity': similarity if similarity is not None else 0.0
                })
        
        # calculate[CN]
        if all_timings:
            print(f"\n=== [CN] ({len(all_timings)} [CN]) ===")
            avg_phases = {}
            for phase in ['phase1', 'phase2', 'phase3', 'phase4', 'total']:
                avg_phases[phase] = np.mean([t[phase] for t in all_timings])
            
            print(f"  [CN]1 ([CN]VDPF[CN]): {avg_phases['phase1']:.2f}[CN]")
            print(f"  [CN]2 (e/fcalculate): {avg_phases['phase2']:.2f}[CN]")
            print(f"  [CN]3 ([CN]): {avg_phases['phase3']:.2f}[CN]")
            print(f"  [CN]4 ([CN]): {avg_phases['phase4']:.2f}[CN]")
            print(f"  [CN]: {avg_phases['total']:.2f}[CN]")
            
            if all_similarities:
                avg_similarity = np.mean(all_similarities)
                print(f"  [CN]: {avg_similarity:.6f}")
            else:
                avg_similarity = 0.0
            
            # [CN]（[CN]--no-report）
            if not args.no_report and query_details:
                report_file = "~/trident/result.md"
                markdown_report = generate_markdown_report(
                    args.dataset, 
                    query_details, 
                    avg_phases,
                    avg_similarity
                )
                
                # [CN]，[CN]，[CN]create[CN]
                if os.path.exists(report_file):
                    # [CN]，[CN]
                    with open(report_file, 'a', encoding='utf-8') as f:
                        f.write("\n\n---\n\n")  # [CN]
                        f.write(markdown_report)
                else:
                    # create[CN]
                    with open(report_file, 'w', encoding='utf-8') as f:
                        f.write(markdown_report)
                
                print(f"\n[CN]{'[CN]' if os.path.exists(report_file) else '[CN]'}[CN]: {report_file}")
            
    except KeyboardInterrupt:
        print("\n[CN]")
    except Exception as e:
        print(f"[CN]: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.disconnect_from_servers()


if __name__ == "__main__":
    main()