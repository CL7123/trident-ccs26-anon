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
import logging
from typing import Dict, List, Optional

# Add project paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('~/trident/src')
sys.path.append('~/trident/standardDPF')
sys.path.append('~/trident/query-opti')

from dpf_wrapper import VDPFVectorWrapper
from dpf_wrapper_optimized import OptimizedVDPFVectorWrapper
from binary_protocol import BinaryProtocol
from basic_functionalities import get_config, MPC23SSS, Share
from share_data import DatasetLoader

# Load configuration
try:
    from config import CLIENT_SERVERS as SERVERS
except ImportError:
    # Default configuration - client uses public IPs
    SERVERS = {
        1: {"host": "192.168.1.101", "port": 8001},
        2: {"host": "192.168.1.102", "port": 8002},
        3: {"host": "192.168.1.103", "port": 8003}
    }

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [Client] %(message)s')
logger = logging.getLogger(__name__)


class DistributedClient:
    """Test client for distributed environment"""
    
    def __init__(self, dataset: str = "siftsmall", servers_config: Dict = None):
        self.dataset = dataset
        self.config = get_config(dataset)
        self.dpf_wrapper = OptimizedVDPFVectorWrapper(dataset_name=dataset)
        self.mpc = MPC23SSS(self.config)
        
        # Preload original data for verification
        data_dir = f"~/trident/dataset/{dataset}"
        loader = DatasetLoader(data_dir)
        self.original_nodes = loader.load_nodes()
        logger.info(f"Preloaded {len(self.original_nodes)} node vectors for verification")
        logger.info(f"Dataset: {dataset}, node data type: {self.original_nodes.dtype}, shape: {self.original_nodes.shape}")

        # Server configuration
        self.servers_config = servers_config or SERVERS
        self.connections = {}
        self.connection_retry_count = 3
        self.connection_timeout = 10

    def connect_to_servers(self):
        """Connect to all servers with retry mechanism"""
        successful_connections = 0

        for server_id, server_info in self.servers_config.items():
            host = server_info["host"]
            port = server_info["port"]

            connected = False
            for attempt in range(self.connection_retry_count):
                try:
                    logger.info(f"Attempting to connect to server {server_id} ({host}:{port}), attempt {attempt + 1}...")

                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(self.connection_timeout)
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                    sock.connect((host, port))
                    self.connections[server_id] = sock
                    logger.info(f"Successfully connected to server {server_id}")
                    successful_connections += 1
                    connected = True
                    break

                except socket.timeout:
                    logger.warning(f"Connection to server {server_id} timed out")
                except ConnectionRefusedError:
                    logger.warning(f"Server {server_id} refused connection")
                except Exception as e:
                    logger.warning(f"Failed to connect to server {server_id}: {e}")

                if attempt < self.connection_retry_count - 1:
                    time.sleep(2)  # Wait before retry

            if not connected:
                logger.error(f"Unable to connect to server {server_id}")

        logger.info(f"Successfully connected to {successful_connections}/{len(self.servers_config)} servers")
        return successful_connections >= 2  # At least 2 servers required
    
    def _send_request(self, server_id: int, request: dict) -> Optional[dict]:
        """Send request to specified server with error handling"""
        if server_id not in self.connections:
            logger.error(f"Not connected to server {server_id}")
            return None

        sock = self.connections[server_id]

        try:
            # Send request with key using binary protocol
            if 'dpf_key' in request:
                # Increase timeout for query requests
                old_timeout = sock.gettimeout()
                sock.settimeout(60)  # 60 second timeout

                BinaryProtocol.send_binary_request(
                    sock,
                    request['command'],
                    request['dpf_key'],
                    request.get('query_id')
                )
                # Receive response
                response = BinaryProtocol.receive_response(sock)

                # Restore original timeout setting
                sock.settimeout(old_timeout)
                return response
            else:
                # Other requests use JSON
                request_data = json.dumps(request).encode()
                sock.sendall(len(request_data).to_bytes(4, 'big'))
                sock.sendall(request_data)

                # Receive response
                length_bytes = sock.recv(4)
                if not length_bytes:
                    raise ConnectionError("Connection closed")

                length = int.from_bytes(length_bytes, 'big')
                data = b''
                while len(data) < length:
                    chunk = sock.recv(min(length - len(data), 4096))
                    if not chunk:
                        raise ConnectionError("Connection interrupted while receiving data")
                    data += chunk

                return json.loads(data.decode())

        except Exception as e:
            logger.error(f"Error communicating with server {server_id}: {e}")
            return None
    
    def test_distributed_query(self, node_id: int = 1723):
        """Test distributed query"""
        # Generate VDPF keys
        keys = self.dpf_wrapper.generate_keys('node', node_id)

        # Generate query ID
        query_id = f'distributed_test_{time.time()}_{node_id}'

        logger.info(f"Starting distributed query, node ID: {node_id}, query ID: {query_id}")

        # Query all servers in parallel
        start_time = time.time()

        def query_server(server_id):
            request = {
                'command': 'query_node_vector',
                'dpf_key': keys[server_id - 1],
                'query_id': query_id
            }
            response = self._send_request(server_id, request)
            return server_id, response

        # Execute queries in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.connections)) as executor:
            futures = [executor.submit(query_server, sid) for sid in self.connections]
            results = {}

            for future in concurrent.futures.as_completed(futures):
                try:
                    server_id, response = future.result()
                    results[server_id] = response
                except Exception as e:
                    logger.error(f"Error querying server: {e}")

        # Check results
        successful_responses = {sid: r for sid, r in results.items()
                              if r and r.get('status') == 'success'}

        if len(successful_responses) < 2:
            logger.error("Query failed: fewer than 2 servers returned successful response")
            for server_id, result in results.items():
                if not result or result.get('status') != 'success':
                    logger.error(f"Server {server_id}: {result}")
            return None, None

        # Extract timing information
        timings = {}
        for server_id, result in successful_responses.items():
            timing = result.get('timing', {})
            timings[server_id] = {
                'phase1': timing.get('phase1_time', 0) / 1000,
                'phase2': timing.get('phase2_time', 0) / 1000,
                'phase3': timing.get('phase3_time', 0) / 1000,
                'phase4': timing.get('phase4_time', 0) / 1000,
                'total': timing.get('total', 0) / 1000
            }

        # Calculate average timings
        avg_timings = {}
        for phase in ['phase1', 'phase2', 'phase3', 'phase4', 'total']:
            phase_times = [t[phase] for t in timings.values()]
            avg_timings[phase] = np.mean(phase_times) if phase_times else 0

        # Reconstruct final result
        final_result = self._reconstruct_final_result(successful_responses)

        # Verify result
        similarity = self._verify_result(node_id, final_result)

        # Print results
        total_time = time.time() - start_time
        logger.info(f"\nQuery results:")
        logger.info(f"  Client total time: {total_time:.2f}s")
        logger.info(f"  Phase 1 (VDPF evaluation): {avg_timings['phase1']:.2f}s")
        logger.info(f"  Phase 2 (e/f calculation): {avg_timings['phase2']:.2f}s")
        logger.info(f"  Phase 3 (data exchange): {avg_timings['phase3']:.2f}s")
        logger.info(f"  Phase 4 (reconstruction): {avg_timings['phase4']:.2f}s")
        logger.info(f"  Server average total: {avg_timings['total']:.2f}s")
        if similarity is not None:
            logger.info(f"  Cosine similarity: {similarity:.6f}")

        return avg_timings, final_result
    
    def _reconstruct_final_result(self, results):
        """Reconstruct final result"""
        # Get responses from at least two servers
        server_ids = sorted([sid for sid, r in results.items()
                           if r and r.get('status') == 'success'])[:2]

        if len(server_ids) < 2:
            logger.error("Reconstruction failed: fewer than 2 available servers")
            return np.zeros(512 if self.dataset == "laion" else 128, dtype=np.float32)

        # Get vector dimension
        first_result = results[server_ids[0]]['result_share']
        vector_dim = len(first_result)

        # Reconstruct each dimension
        reconstructed_vector = np.zeros(vector_dim, dtype=np.float32)

        # Scale factor
        if self.dataset == "siftsmall":
            scale_factor = 1048576  # 2^20
        else:
            scale_factor = 536870912  # 2^29

        for i in range(vector_dim):
            shares = [
                Share(results[server_ids[0]]['result_share'][i], server_ids[0]),
                Share(results[server_ids[1]]['result_share'][i], server_ids[1])
            ]

            reconstructed = self.mpc.reconstruct(shares)

            # Convert back to floating point
            if reconstructed > self.config.prime // 2:
                signed = reconstructed - self.config.prime
            else:
                signed = reconstructed

            reconstructed_vector[i] = signed / scale_factor

        return reconstructed_vector
    
    def _verify_result(self, node_id: int, reconstructed_vector: np.ndarray):
        """Verify correctness of reconstructed result"""
        try:
            if node_id < len(self.original_nodes):
                original_vector = self.original_nodes[node_id]

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

        except Exception as e:
            logger.error(f"Error verifying result: {e}")
            return None
    
    def get_server_status(self):
        """Get status of all servers"""
        logger.info("Getting server status...")

        for server_id in self.connections:
            request = {'command': 'get_status'}
            response = self._send_request(server_id, request)

            if response and response.get('status') == 'success':
                logger.info(f"\nServer {server_id} status:")
                logger.info(f"  Mode: {response.get('mode')}")
                logger.info(f"  Address: {response.get('host')}:{response.get('port')}")
                logger.info(f"  Dataset: {response.get('dataset')}")
                logger.info(f"  VDPF processes: {response.get('vdpf_processes')}")
                logger.info(f"  Data loaded: {response.get('data_loaded')}")
                logger.info(f"  Triples available: {response.get('triples_available')}")
            else:
                logger.error(f"Unable to get status of server {server_id}")
    
    def disconnect_from_servers(self):
        """Disconnect from all servers"""
        for server_id, sock in self.connections.items():
            try:
                sock.close()
                logger.info(f"Disconnected from server {server_id}")
            except:
                pass
        self.connections.clear()


def generate_markdown_report(dataset, query_details, avg_phases, avg_similarity):
    """Generate Markdown format test report"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    markdown = f"""# Distributed Test Report - {dataset}

**Generated time**: {timestamp}
**Dataset**: {dataset}
**Number of queries**: {len(query_details)}

## Detailed Query Results

| Query # | Node ID | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruct) | Total Time | Cosine Similarity |
|---------|---------|----------------|---------------|--------------------|----------------------|------------|-------------------|
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

- **Phase 1 (VDPF evaluation)**: {avg_phases['phase1']:.2f}s
- **Phase 2 (e/f calculation)**: {avg_phases['phase2']:.2f}s
- **Phase 3 (data exchange)**: {avg_phases['phase3']:.2f}s
- **Phase 4 (reconstruction)**: {avg_phases['phase4']:.2f}s
- **Server average total**: {avg_phases['total']:.2f}s
- **Average cosine similarity**: {avg_similarity:.6f}

## Performance Analysis

### Time Distribution
"""

    # Calculate time proportion for each phase
    total_avg = avg_phases['total']
    if total_avg > 0:
        phase1_pct = (avg_phases['phase1'] / total_avg) * 100
        phase2_pct = (avg_phases['phase2'] / total_avg) * 100
        phase3_pct = (avg_phases['phase3'] / total_avg) * 100
        phase4_pct = (avg_phases['phase4'] / total_avg) * 100

        markdown += f"""
- Phase 1 (VDPF evaluation): {phase1_pct:.1f}%
- Phase 2 (e/f calculation): {phase2_pct:.1f}%
- Phase 3 (data exchange): {phase3_pct:.1f}%
- Phase 4 (reconstruction): {phase4_pct:.1f}%

### Throughput
- Average query time: {total_avg:.2f}s
- Theoretical throughput: {1/total_avg:.2f} queries/second
"""

    return markdown


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Distributed vector query client')
    parser.add_argument('--dataset', type=str, default='siftsmall',
                        choices=['siftsmall', 'laion', 'tripclick', 'ms_marco', 'nfcorpus'],
                        help='Dataset name (default: siftsmall)')
    parser.add_argument('--num-queries', type=int, default=10,
                        help='Number of test queries (default: 10)')
    parser.add_argument('--no-report', action='store_true',
                        help='Do not save test report')
    parser.add_argument('--config', type=str,
                        help='Path to server configuration file')
    parser.add_argument('--status-only', action='store_true',
                        help='Only get server status')

    args = parser.parse_args()

    # Load custom configuration
    servers_config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                servers_config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load configuration file: {e}")
            return

    logger.info(f"=== Distributed Test Client - Dataset: {args.dataset} ===")

    client = DistributedClient(args.dataset, servers_config)

    try:
        # Connect to servers
        if not client.connect_to_servers():
            logger.error("Failed to connect to servers")
            return

        # If only getting status
        if args.status_only:
            client.get_server_status()
            return

        all_timings = []
        all_similarities = []
        query_details = []

        # Get total number of nodes
        total_nodes = len(client.original_nodes)

        # Randomly select nodes
        random_nodes = random.sample(range(total_nodes), min(args.num_queries, total_nodes))

        logger.info(f"Will perform query tests on {len(random_nodes)} random nodes...\n")

        for i, node_id in enumerate(random_nodes):
            logger.info(f"Query {i+1}/{len(random_nodes)}: node {node_id}")
            timings, final_result = client.test_distributed_query(node_id=node_id)

            if timings:
                all_timings.append(timings)
                similarity = client._verify_result(node_id, final_result)
                if similarity is not None:
                    all_similarities.append(similarity)

                query_details.append({
                    'query_num': i + 1,
                    'node_id': node_id,
                    'timings': timings,
                    'similarity': similarity if similarity is not None else 0.0
                })

        # Calculate average values
        if all_timings:
            logger.info(f"\n=== Average Performance Statistics ({len(all_timings)} successful queries) ===")
            avg_phases = {}
            for phase in ['phase1', 'phase2', 'phase3', 'phase4', 'total']:
                avg_phases[phase] = np.mean([t[phase] for t in all_timings])

            logger.info(f"  Phase 1 (VDPF evaluation): {avg_phases['phase1']:.2f}s")
            logger.info(f"  Phase 2 (e/f calculation): {avg_phases['phase2']:.2f}s")
            logger.info(f"  Phase 3 (data exchange): {avg_phases['phase3']:.2f}s")
            logger.info(f"  Phase 4 (reconstruction): {avg_phases['phase4']:.2f}s")
            logger.info(f"  Server average total: {avg_phases['total']:.2f}s")

            if all_similarities:
                avg_similarity = np.mean(all_similarities)
                logger.info(f"  Average cosine similarity: {avg_similarity:.6f}")
            else:
                avg_similarity = 0.0

            # Save report
            if not args.no_report and query_details:
                report_file = "~/trident/distributed-deploy/distributed_result.md"
                markdown_report = generate_markdown_report(
                    args.dataset,
                    query_details,
                    avg_phases,
                    avg_similarity
                )

                # Append mode with separator
                with open(report_file, 'a', encoding='utf-8') as f:
                    # If file exists and is not empty, add separator
                    f.seek(0, 2)  # Move to end of file
                    if f.tell() > 0:
                        f.write("\n\n---\n\n")
                    f.write(markdown_report)

                logger.info(f"\nTest report saved to: {report_file}")

    except KeyboardInterrupt:
        logger.info("\nUser interrupted")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.disconnect_from_servers()


if __name__ == "__main__":
    main()