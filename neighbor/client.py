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

from dpf_wrapper import VDPFVectorWrapper
from basic_functionalities import get_config, MPC23SSS, Share
from share_data import DatasetLoader


class MultiprocessConfigurableTestClient:
    """[CN]"""
    
    def __init__(self, dataset: str = "laion"):
        self.dataset = dataset
        self.config = get_config(dataset)
        self.dpf_wrapper = VDPFVectorWrapper(dataset_name=dataset)
        self.mpc = MPC23SSS(self.config)
        
        # [CN]
        data_dir = f"~/trident/dataset/{dataset}"
        loader = DatasetLoader(data_dir)
        self.original_nodes, self.num_layers, self.max_neighbors = loader.load_neighbors()
        total_neighbor_entries = len(self.original_nodes) * self.num_layers
        print(f"[CN] {len(self.original_nodes)} [CN]")
        print(f"  [CN] {total_neighbor_entries} [CN]（{len(self.original_nodes)} [CN] × {self.num_layers} [CN]）")
        
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
    
    def test_mmap_query(self, node_id: int = 1723, layer: int = 0):
        """[CN]"""
        # [CN]VDPF[CN] - [CN]node_id*3 + layercalculate[CN]
        neighbor_index = node_id * 3 + layer
        print(f"\nDEBUG: [CN]VDPF[CN] - node_id={node_id}, layer={layer}, neighbor_index={neighbor_index}")
        # [CN]dpf_wrapper[CN]generate_keys[CN]node_id[CN]alpha，[CN]neighbor_index
        keys = self.dpf_wrapper.generate_keys('neighbor', neighbor_index)
        
        # [CN]
        start_time = time.time()
        
        def query_server(server_id):
            request = {
                'command': 'query_node_vector',
                'dpf_key': keys[server_id - 1],
                'query_id': f'multiprocess_test_{int(time.time())}',
                'neighbor_index': neighbor_index,
                'node_id': node_id,
                'layer': layer
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
            
            # DEBUG: print[CN]
            print(f"\nDEBUG: [CN]:")
            for sid, result in results.items():
                if result.get('status') == 'success':
                    shares = result['result_share']
                    print(f"  Server {sid} [CN]5[CN]: {shares[:5]}")
            
            # [CN]
            match_ratio = self._verify_result(node_id, layer, final_result)
            
            # [CN]print[CN]
            print(f"\n[CN] ([CN]{node_id}, [CN]{layer}):")
            print(f"  [CN]1 ([CN]VDPF[CN]): {avg_timings['phase1']:.2f}[CN]")
            print(f"  [CN]2 (e/fcalculate): {avg_timings['phase2']:.2f}[CN]")
            print(f"  [CN]3 ([CN]): {avg_timings['phase3']:.2f}[CN]")
            print(f"  [CN]4 ([CN]): {avg_timings['phase4']:.2f}[CN]")
            print(f"  [CN]: {avg_timings['total']:.2f}[CN]")
            if match_ratio is not None:
                print(f"  [CN]: {match_ratio:.2%}")
            
            return avg_timings, final_result
            
        else:
            print("❌ [CN]:")
            for server_id, result in results.items():
                if result.get('status') != 'success':
                    print(f"  MultiProcess Server {server_id}: {result.get('message', 'Unknown error')}")
            return None, None
    
    def _reconstruct_final_result(self, results):
        """[CN] - [CN]"""
        # [CN]servers[CN]
        server_ids = []
        for server_id in [1, 2, 3]:
            if server_id in results and results[server_id].get('status') == 'success':
                server_ids.append(server_id)
        
        if len(server_ids) < 2:
            print("[CN]：[CN]2servers[CN]")
            return []
        
        # [CN]
        server_ids = sorted(server_ids)[:2]
        print(f"DEBUG: [CN] {server_ids} [CN]")
        
        # [CN]
        first_result = results[server_ids[0]]['result_share']
        num_neighbors = len(first_result)
        
        # [CN]ID
        reconstructed_neighbors = []
        
        for i in range(num_neighbors):
            shares = [
                Share(results[server_ids[0]]['result_share'][i], server_ids[0]),
                Share(results[server_ids[1]]['result_share'][i], server_ids[1])
            ]
            
            reconstructed = self.mpc.reconstruct(shares)
            
            # DEBUG: print[CN]
            if i < 5:  # [CN]print[CN]5[CN]
                print(f"  DEBUG: [CN]{i} - [CN]: [{shares[0].value}, {shares[1].value}], [CN]: {reconstructed}")
                
                # [CN]：[CN]
                if i == 0:
                    # create[CN]
                    test_value = 4420  # [CN]
                    test_shares = self.mpc.share_secret(test_value)
                    test_reconstructed = self.mpc.reconstruct([test_shares[0], test_shares[1]])
                    print(f"    TEST: [CN] {test_value} -> [CN] [{test_shares[0].value}, {test_shares[1].value}] -> [CN] {test_reconstructed}")
            
            # [CN]ID[CN]，[CN]
            reconstructed_neighbors.append(reconstructed)
        
        # DEBUG: print[CN]
        print(f"\nDEBUG: [CN]:")
        print(f"  [CN]: {len(reconstructed_neighbors)}")
        # [CN]
        valid_neighbors = [n for n in reconstructed_neighbors if n != self.config.prime - 1 and n < len(self.original_nodes)]
        print(f"  [CN]: {len(valid_neighbors)}")
        if valid_neighbors:
            print(f"  [CN]10[CN]: {valid_neighbors[:10]}")
        
        return reconstructed_neighbors
    
    def _verify_result(self, node_id: int, layer: int, reconstructed_neighbors: list):
        """[CN]，return[CN]"""
        try:
            if node_id < len(self.original_nodes) and layer < self.num_layers:
                original_neighbors = self.original_nodes[node_id][layer]
                
                # DEBUG: print[CN]
                print(f"\nDEBUG: [CN] ([CN] {node_id}, [CN] {layer}):")
                print(f"  [CN]: {len(original_neighbors)}")
                print(f"  [CN]10[CN]: {original_neighbors[:10]}")
                
                # [CN]（-1[CN]field_size-1）
                reconstructed_filtered = [n for n in reconstructed_neighbors if n != self.config.prime - 1 and n < len(self.original_nodes)]
                
                # calculate[CN]
                if len(original_neighbors) == 0:
                    return 1.0 if len(reconstructed_filtered) == 0 else 0.0
                
                # calculate[CN]
                matches = len(set(original_neighbors) & set(reconstructed_filtered))
                match_ratio = matches / len(original_neighbors)
                
                print(f"  [CN]: {len(reconstructed_filtered)}")
                print(f"  [CN]: {matches}/{len(original_neighbors)}")
                print(f"  [CN]: {match_ratio:.2%}")
                
                return match_ratio
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

| [CN] | [CN]ID | [CN] | [CN]1 (VDPF) | [CN]2 (e/f) | [CN]3 ([CN]) | [CN]4 ([CN]) | [CN] | [CN] |
|---------|--------|-----|--------------|-------------|--------------|--------------|--------|-----------|
"""
    
    for q in query_details:
        markdown += f"| {q['query_num']} | {q['node_id']} | {q.get('layer', 0)} | "
        markdown += f"{q['timings']['phase1']:.2f}s | "
        markdown += f"{q['timings']['phase2']:.2f}s | "
        markdown += f"{q['timings']['phase3']:.2f}s | "
        markdown += f"{q['timings']['phase4']:.2f}s | "
        markdown += f"{q['timings']['total']:.2f}s | "
        markdown += f"{q['similarity']:.2%} |\n"
    
    markdown += f"""
## [CN]

- **[CN]1 ([CN]VDPF[CN])**: {avg_phases['phase1']:.2f}[CN]
- **[CN]2 (e/fcalculate)**: {avg_phases['phase2']:.2f}[CN]
- **[CN]3 ([CN])**: {avg_phases['phase3']:.2f}[CN]
- **[CN]4 ([CN])**: {avg_phases['phase4']:.2f}[CN]
- **[CN]**: {avg_phases['total']:.2f}[CN]
- **[CN]**: {avg_similarity:.2%}

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
            # [CN]
            layer = random.randint(0, 2)  # [CN]3[CN]
            print(f"[CN] {i+1}/{len(random_nodes)}: [CN] {node_id}, [CN] {layer}")
            timings, final_result = client.test_mmap_query(node_id=node_id, layer=layer)
            
            if timings:
                all_timings.append(timings)
                # [CN]
                match_ratio = client._verify_result(node_id, layer, final_result)
                if match_ratio is not None:
                    all_similarities.append(match_ratio)
                
                # [CN]
                query_details.append({
                    'query_num': i + 1,
                    'node_id': node_id,
                    'layer': layer,
                    'timings': timings,
                    'similarity': match_ratio if match_ratio is not None else 0.0
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
                print(f"  [CN]: {avg_similarity:.2%}")
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