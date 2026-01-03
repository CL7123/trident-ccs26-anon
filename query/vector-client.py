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
    """Test client for multiprocess optimization server"""
    
    def __init__(self, dataset: str = "laion"):
        self.dataset = dataset
        self.config = get_config(dataset)
        self.dpf_wrapper = VDPFVectorWrapper(dataset_name=dataset)
        self.mpc = MPC23SSS(self.config)
        
        # Preload original data for validation
        data_dir = f"~/trident/dataset/{dataset}"
        loader = DatasetLoader(data_dir)
        self.original_nodes = loader.load_nodes()
        print(f"Preloaded {len(self.original_nodes)} node vectors for verification")
        
        # serveraddress
        self.servers = [
            ("192.168.50.21", 8001),
            ("192.168.50.22", 8002),
            ("192.168.50.23", 8003)
        ]
        
        self.connections = {}
        
    def connect_to_servers(self):
        """Connect to all multiprocess optimization servers"""
        for i, (host, port) in enumerate(self.servers):
            server_id = i + 1
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((host, port))
                self.connections[server_id] = sock
            except Exception:
                pass
        
    def _send_request(self, server_id: int, request: dict) -> dict:
        """Send request to specified server"""
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
    
    def test_mmap_query(self, node_id: int = 1723):
        """Test multiprocess optimization query"""
        # Generate VDPF keys
        keys = self.dpf_wrapper.generate_keys('node', node_id)
        
        # Parallel query to all servers
        start_time = time.time()
        
        def query_server(server_id):
            request = {
                'command': 'query_node_vector',
                'dpf_key': keys[server_id - 1],
                'query_id': f'multiprocess_test_{int(time.time())}'
            }
            response = self._send_request(server_id, request)
            return server_id, response
        
        # Execute queries in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(query_server, sid) for sid in self.connections]
            results = {}
            
            for future in concurrent.futures.as_completed(futures):
                server_id, response = future.result()
                results[server_id] = response
        
        if all(r.get('status') == 'success' for r in results.values()):
            # Extract timing information
            timings = {}
            for server_id, result in results.items():
                timing = result.get('timing', {})
                timings[server_id] = {
                    'phase1': timing.get('phase1_time', 0) / 1000,  # Convert to seconds
                    'phase2': timing.get('phase2_time', 0) / 1000,
                    'phase3': timing.get('phase3_time', 0) / 1000,
                    'phase4': timing.get('phase4_time', 0) / 1000,
                    'total': timing.get('total', 0) / 1000
                }
            
            # Calculate average timings
            avg_timings = {}
            for phase in ['phase1', 'phase2', 'phase3', 'phase4', 'total']:
                avg_timings[phase] = np.mean([t[phase] for t in timings.values()])
            
            # Reconstruct final result - neighbor list
            final_result = self._reconstruct_final_result(results)
            
            # DEBUG: Print share information before reconstruction
            print(f"\nDEBUG: Share information before reconstruction:")
            for sid, result in results.items():
                if result.get('status') == 'success':
                    shares = result['result_share']
                    print(f"  Server {sid} shares first 5 values: {shares[:5]}")
            
            # Verify result correctness and get similarity
            similarity = self._verify_result(node_id, final_result)
            
            # Print core information only
            print(f"\nQuery result:")
            print(f"  Phase 1 (Multiprocess VDPF evaluation): {avg_timings['phase1']:.2f}s")
            print(f"  Phase 2 (e/f computation): {avg_timings['phase2']:.2f}s")
            print(f"  Phase 3 (Data exchange): {avg_timings['phase3']:.2f}s")
            print(f"  Phase 4 (Reconstruction): {avg_timings['phase4']:.2f}s")
            print(f"  Server Internal Total: {avg_timings['total']:.2f}s")
            if similarity is not None:
                print(f"  Cosine similarity: {similarity:.6f}")
            
            return avg_timings, final_result
            
        else:
            print("❌ Query failed:")
            for server_id, result in results.items():
                if result.get('status') != 'success':
                    print(f"  MultiProcess Server {server_id}: {result.get('message', 'Unknown error')}")
            return None, None
    
    def _reconstruct_final_result(self, results):
        """Reconstruct final result - neighbor list"""
        # Get responses from at least two servers
        server_ids = []
        for server_id in [1, 2, 3]:
            if server_id in results and results[server_id].get('status') == 'success':
                server_ids.append(server_id)
        
        if len(server_ids) < 2:
            print("Error: At least 2 server responses are required for reconstruction")
            # Set default dimension according to dataset
            default_dim = 512 if self.dataset == "laion" else 128
            return np.zeros(default_dim, dtype=np.float32)

        # Use first two available servers for reconstruction
        server_ids = sorted(server_ids)[:2]

        # Dynamically get vector dimension
        first_result = results[server_ids[0]]['result_share']
        vector_dim = len(first_result)

        # Reconstruct each dimension
        reconstructed_vector = np.zeros(vector_dim, dtype=np.float32)
        
        # Use scaling factor consistent with secret sharing
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
            
            # Convert back to float
            if reconstructed > self.config.prime // 2:
                signed = reconstructed - self.config.prime
            else:
                signed = reconstructed
            
            reconstructed_vector[i] = signed / scale_factor
        
        # DEBUG: Print reconstructed vector information
        print(f"\nDEBUG: Reconstructed vector information:")
        print(f"  First 5 values: {reconstructed_vector[:5]}")
        print(f"  Range: [{np.min(reconstructed_vector):.6f}, {np.max(reconstructed_vector):.6f}]")
        print(f"  Mean: {np.mean(reconstructed_vector):.6f}, Standard deviation: {np.std(reconstructed_vector):.6f}")

        return reconstructed_vector

    def _verify_result(self, node_id: int, reconstructed_vector: np.ndarray):
        """Validate reconstruction result correctness and return similarity"""
        try:
            if node_id < len(self.original_nodes):
                original_vector = self.original_nodes[node_id]

                # DEBUG: Print original vector information
                print(f"\nDEBUG: Original vector information (Node {node_id}):")
                print(f"  First 5 values: {original_vector[:5]}")
                print(f"  Range: [{np.min(original_vector):.6f}, {np.max(original_vector):.6f}]")
                print(f"  Mean: {np.mean(original_vector):.6f}, Standard deviation: {np.std(original_vector):.6f}")

                # Calculate cosine similarity
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
        """Disconnect from all servers"""
        for sock in self.connections.values():
            sock.close()
        self.connections.clear()


def generate_markdown_report(dataset, query_details, avg_phases, avg_similarity):
    """Generate Markdown format test report"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    markdown = f"""# Test Results Report - {dataset}

**Generation Time**: {timestamp}
**Dataset**: {dataset}
**Number of Queries**: {len(query_details)}

## Detailed Query Results

| Query # | NodeID | Phase1 (VDPF) | Phase2 (e/f) | Phase3 (Exchange) | Phase4 (Reconstruction) | Total Time | Cosine Similarity |
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
## Average Performance Statistics

- **Phase 1 (Multiprocess VDPF evaluation)**: {avg_phases['phase1']:.2f}s
- **Phase 2 (e/f computation)**: {avg_phases['phase2']:.2f}s
- **Phase 3 (Data exchange)**: {avg_phases['phase3']:.2f}s
- **Phase 4 (Reconstruction)**: {avg_phases['phase4']:.2f}s
- **Server Internal Total**: {avg_phases['total']:.2f}s
- **Average Cosine Similarity**: {avg_similarity:.6f}

## Performance Analysis

### Time Distribution
"""

    # Calculate time percentage for each phase
    total_avg = avg_phases['total']
    phase1_pct = (avg_phases['phase1'] / total_avg) * 100
    phase2_pct = (avg_phases['phase2'] / total_avg) * 100
    phase3_pct = (avg_phases['phase3'] / total_avg) * 100
    phase4_pct = (avg_phases['phase4'] / total_avg) * 100

    markdown += f"""
- Phase 1 (VDPF evaluation): {phase1_pct:.1f}%
- Phase 2 (e/f computation): {phase2_pct:.1f}%
- Phase 3 (Data exchange): {phase3_pct:.1f}%
- Phase 4 (Reconstruction): {phase4_pct:.1f}%

### Throughput
- Average Query Time: {total_avg:.2f}s
- Theoretical Throughput: {1/total_avg:.2f} queries/sec
"""
    
    return markdown


def main():
    """Main function"""
    # Set command-line arguments
    parser = argparse.ArgumentParser(description='Vector-level multiprocess optimization client')
    parser.add_argument('--dataset', type=str, default='siftsmall', 
                        choices=['siftsmall', 'laion', 'tripclick', 'ms_marco', 'nfcorpus'],
                        help='Dataset name (default: siftsmall)')
    parser.add_argument('--num-queries', type=int, default=10,
                        help='Number of test queries (default: 10)')
    parser.add_argument('--no-report', action='store_true',
                        help='Do not save test report')
    
    args = parser.parse_args()
    
    print(f"=== Multiprocess Configuration Test - Dataset: {args.dataset} ===")

    client = MultiprocessConfigurableTestClient(args.dataset)

    try:
        client.connect_to_servers()

        if len(client.connections) == 0:
            print("Error: Unable to connect to any multiprocess server")
            return

        all_timings = []
        all_similarities = []
        query_details = []  # Store details for each query

        # Get total number of nodes
        total_nodes = len(client.original_nodes)

        # Randomly select nodes
        random_nodes = random.sample(range(total_nodes), min(args.num_queries, total_nodes))

        print(f"Will test queries on {len(random_nodes)} randomly selected nodes...\n")
        
        for i, node_id in enumerate(random_nodes):
            print(f"Query {i+1}/{len(random_nodes)}: Node {node_id}")
            timings, final_result = client.test_mmap_query(node_id=node_id)
            
            if timings:
                all_timings.append(timings)
                # Get similarity
                similarity = client._verify_result(node_id, final_result)
                if similarity is not None:
                    all_similarities.append(similarity)
                
                # Save query details
                query_details.append({
                    'query_num': i + 1,
                    'node_id': node_id,
                    'timings': timings,
                    'similarity': similarity if similarity is not None else 0.0
                })

        # Calculate average
        if all_timings:
            print(f"\n=== Average Performance Statistics ({len(all_timings)} successful queries) ===")
            avg_phases = {}
            for phase in ['phase1', 'phase2', 'phase3', 'phase4', 'total']:
                avg_phases[phase] = np.mean([t[phase] for t in all_timings])

            print(f"  Phase 1 (Multiprocess VDPF evaluation): {avg_phases['phase1']:.2f}s")
            print(f"  Phase 2 (e/f computation): {avg_phases['phase2']:.2f}s")
            print(f"  Phase 3 (Data exchange): {avg_phases['phase3']:.2f}s")
            print(f"  Phase 4 (Reconstruction): {avg_phases['phase4']:.2f}s")
            print(f"  Server Internal Total: {avg_phases['total']:.2f}s")

            if all_similarities:
                avg_similarity = np.mean(all_similarities)
                print(f"  Average Cosine Similarity: {avg_similarity:.6f}")
            else:
                avg_similarity = 0.0

            # Save report (unless --no-report is specified)
            if not args.no_report and query_details:
                report_file = "~/trident/result.md"
                markdown_report = generate_markdown_report(
                    args.dataset, 
                    query_details, 
                    avg_phases,
                    avg_similarity
                )
                
                # Check if file exists, append if exists, create new if not
                file_exists = os.path.exists(report_file)
                if file_exists:
                    # Append mode, add separator first
                    with open(report_file, 'a', encoding='utf-8') as f:
                        f.write("\n\n---\n\n")  # Add separator
                        f.write(markdown_report)
                else:
                    # Create new file
                    with open(report_file, 'w', encoding='utf-8') as f:
                        f.write(markdown_report)

                print(f"\nTest report {'appended to' if file_exists else 'saved to'}: {report_file}")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.disconnect_from_servers()


if __name__ == "__main__":
    main()