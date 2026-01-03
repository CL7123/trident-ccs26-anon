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
    # Default configuration - client uses public IP
    SERVERS = {
        1: {"host": "192.168.1.101", "port": 8001},
        2: {"host": "192.168.1.102", "port": 8002},
        3: {"host": "192.168.1.103", "port": 8003}
    }

# Set logging
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
        logger.info(f"Preloaded {len(self.original_nodes)}  node vectors for verification")
        logger.info(f"Dataset: {dataset}, Node data type: {self.original_nodes.dtype}, Shape: {self.original_nodes.shape}")
        
        # Serverconfig
        self.servers_config = servers_config or SERVERS
        self.connections = {}
        self.connection_retry_count = 3
        self.connection_timeout = 10
        
    def connect_to_servers(self):
        """Connect to all servers with retry mechanism"""
        successful_connections = 0 # number of successful connections
        
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
                    logger.warning(f"Server {server_id} connection timeout")
                except ConnectionRefusedError:
                    logger.warning(f"Server {server_id} refused connection")
                except Exception as e:
                    logger.warning(f"Server {server_id} connection failed: {e}")

                if attempt < self.connection_retry_count - 1:
                    time.sleep(2)  # Wait before retry

            if not connected:
                logger.error(f"Unable to connect to server {server_id}")

        logger.info(f"Successfully connected to {successful_connections}/{len(self.servers_config)} servers")
        return successful_connections >= 2  # Need at least 2 servers

    def _send_request(self, server_id: int, request: dict) -> Optional[dict]:
        """Send request to specified server with error handling"""
        if server_id not in self.connections:
            logger.error(f"Not connected to server {server_id}")
            return None

        sock = self.connections[server_id]

        try:
            # Use binary protocol to send requests with keys
            if 'dpf_key' in request:
                # For query requests, increase timeout
                old_timeout = sock.gettimeout()
                sock.settimeout(60)  # 60 seconds timeout

                BinaryProtocol.send_binary_request(
                    sock,
                    request['command'],
                    request['dpf_key'],
                    request.get('query_id')
                )
                # Receive response
                response = BinaryProtocol.receive_response(sock)

                # Restore original timeout settings
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

    def _send_request_with_stats(self, server_id: int, request: dict) -> tuple:
        """
        Send request to specified server and collect data statistics

        Returns:
            tuple: (response, stats_dict) where stats_dict contains:
                - bytes_sent: Number of bytes sent
                - bytes_received: Number of bytes received
                - send_duration_ms: Send duration
                - receive_duration_ms: Receive duration
        """
        if server_id not in self.connections:
            logger.error(f"Not connected to server {server_id}")
            return None, {'error': 'not_connected'}

        sock = self.connections[server_id]
        stats = {
            'bytes_sent': 0,
            'bytes_received': 0,
            'send_duration_ms': 0,
            'receive_duration_ms': 0
        }

        try:
            # Use binary protocol to send requests with keys
            if 'dpf_key' in request:
                # For query requests, increase timeout
                old_timeout = sock.gettimeout()
                sock.settimeout(60)  # 60 seconds timeout

                # Calculate size of data being sent
                dpf_key = request['dpf_key']
                query_id = request.get('query_id', '')
                command = request['command']

                # Estimate data being sent (consistent with BinaryProtocol.send_binary_request)
                import pickle
                key_bytes = pickle.dumps(dpf_key)
                metadata = {
                    'command': command,
                    'query_id': query_id,
                    'key_size': len(key_bytes)
                }
                metadata_bytes = json.dumps(metadata).encode()
                stats['bytes_sent'] = 4 + len(metadata_bytes) + 4 + len(key_bytes)

                # Send request
                send_start = time.time()
                BinaryProtocol.send_binary_request(sock, command, dpf_key, query_id)
                stats['send_duration_ms'] = (time.time() - send_start) * 1000

                # Receive response (manually receive to count actual bytes)
                receive_start = time.time()

                # Read length header (4 bytes)
                import struct
                length_data = sock.recv(4)
                if not length_data:
                    raise ConnectionError("Failed to receive response length")

                total_len = struct.unpack('>I', length_data)[0]

                # Read complete response data
                data = b''
                while len(data) < total_len:
                    chunk = sock.recv(min(4096, total_len - len(data)))
                    if not chunk:
                        raise ConnectionError("Connection interrupted while receiving response data")
                    data += chunk

                # Parse response
                response = json.loads(data.decode('utf-8'))

                stats['receive_duration_ms'] = (time.time() - receive_start) * 1000
                stats['bytes_received'] = 4 + len(data)  # Actual bytes received

                # Restore original timeout settings
                sock.settimeout(old_timeout)
                return response, stats
            else:
                # Other requests use JSON
                request_data = json.dumps(request).encode()
                stats['bytes_sent'] = 4 + len(request_data)

                send_start = time.time()
                sock.sendall(len(request_data).to_bytes(4, 'big'))
                sock.sendall(request_data)
                stats['send_duration_ms'] = (time.time() - send_start) * 1000

                # Receive response
                receive_start = time.time()
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

                stats['receive_duration_ms'] = (time.time() - receive_start) * 1000
                stats['bytes_received'] = 4 + len(data)

                return json.loads(data.decode()), stats

        except Exception as e:
            logger.error(f"Error communicating with server {server_id}: {e}")
            stats['error'] = str(e)
            return None, stats
    
    def test_distributed_query(self, node_id: int = 1723):
        """Test distributed query"""
        # Generate VDPF keys
        keys = self.dpf_wrapper.generate_keys('node', node_id)

        # Generate query ID
        query_id = f'distributed_test_{time.time()}_{node_id}'

        logger.info(f"Starting distributed query, Node ID: {node_id}, Query ID: {query_id}")

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

        # Execute query in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.connections)) as executor:
            futures = [executor.submit(query_server, sid) for sid in self.connections]
            results = {}

            for future in concurrent.futures.as_completed(futures):
                try:
                    server_id, response = future.result()
                    results[server_id] = response
                except Exception as e:
                    logger.error(f"Error querying server: {e}")
        
        # Check result
        successful_responses = {sid: r for sid, r in results.items() 
                              if r and r.get('status') == 'success'}
        
        if len(successful_responses) < 2:
            logger.error("Query failed: fewer than 2 servers with successful responses")
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
        
        # Calculate average timing
        avg_timings = {}
        for phase in ['phase1', 'phase2', 'phase3', 'phase4', 'total']:
            phase_times = [t[phase] for t in timings.values()]
            avg_timings[phase] = np.mean(phase_times) if phase_times else 0
        
        # Reconstruct final result
        final_result = self._reconstruct_final_result(successful_responses)
        
        # Verify result
        similarity = self._verify_result(node_id, final_result)
        
        # Print result
        total_time = time.time() - start_time
        logger.info(f"\nQuery results:")
        logger.info(f"  Client total time: {total_time:.2f}seconds")
        logger.info(f"  Phase 1 (VDPF Evaluation): {avg_timings['phase1']:.2f}seconds")
        logger.info(f"  Phase 2 (e/f Computation): {avg_timings['phase2']:.2f}seconds")
        logger.info(f"  Phase 3 (Data Exchange): {avg_timings['phase3']:.2f}seconds")
        logger.info(f"  Phase 4 (Reconstruction): {avg_timings['phase4']:.2f}seconds")
        logger.info(f"  Server average total: {avg_timings['total']:.2f}seconds")
        if similarity is not None:
            logger.info(f"  Cosine similarity: {similarity:.6f}")

        return avg_timings, final_result

    def detailed_query_with_profiling(self, node_id: int = 1723):
        """
        Execute a single query and collect detailed performance data

        Returns:
            dict: {
                'client_metrics': {
                    'dpf_key_generation': {...},
                    'per_server_communication': {...}
                },
                'server_responses': {server_id: response_data},
                'success': bool
            }
        """
        profiling_data = {
            'query_id': f'profiling_{time.time()}_{node_id}',
            'node_id': node_id,
            'client_metrics': {},
            'server_responses': {},
            'success': False
        }

        # 1. Measure DPF key generation
        logger.info("Phase: DPF Key Generation")
        dpf_gen_start = time.time()
        keys = self.dpf_wrapper.generate_keys('node', node_id)
        dpf_gen_end = time.time()

        profiling_data['client_metrics']['dpf_key_generation'] = {
            'start_time': dpf_gen_start,
            'end_time': dpf_gen_end,
            'duration_ms': (dpf_gen_end - dpf_gen_start) * 1000
        }
        logger.info(f"  DPF key generation: {(dpf_gen_end - dpf_gen_start) * 1000:.2f} ms")

        # Generate query ID
        query_id = profiling_data['query_id']

        # 2. Measure per-server communication
        logger.info("Phase: Server Communication")
        per_server_metrics = {}

        def query_server_with_profiling(server_id):
            server_metrics = {
                'server_id': server_id,
                'connection_start': time.time()
            }

            # Prepare request
            request = {
                'command': 'query_node_vector',
                'dpf_key': keys[server_id - 1],
                'query_id': query_id
            }

            # Send request and receive response with stats
            try:
                overall_start = time.time()
                response, comm_stats = self._send_request_with_stats(server_id, request)
                overall_end = time.time()

                # Merge communication stats
                server_metrics.update(comm_stats)
                server_metrics['rtt_ms'] = (overall_end - overall_start) * 1000
                server_metrics['success'] = response is not None and response.get('status') == 'success'

                # Calculate speeds
                if server_metrics['success']:
                    if comm_stats.get('send_duration_ms', 0) > 0:
                        server_metrics['upload_speed_mbps'] = (comm_stats['bytes_sent'] * 8) / (comm_stats['send_duration_ms'] * 1000)
                    if comm_stats.get('receive_duration_ms', 0) > 0:
                        server_metrics['download_speed_mbps'] = (comm_stats['bytes_received'] * 8) / (comm_stats['receive_duration_ms'] * 1000)

                logger.info(f"  Server {server_id}:")
                logger.info(f"    RTT: {server_metrics['rtt_ms']:.2f} ms")
                logger.info(f"    Sent: {comm_stats.get('bytes_sent', 0)} bytes in {comm_stats.get('send_duration_ms', 0):.2f} ms")
                logger.info(f"    Received: {comm_stats.get('bytes_received', 0)} bytes in {comm_stats.get('receive_duration_ms', 0):.2f} ms")

                return server_id, server_metrics, response

            except Exception as e:
                logger.error(f"  Server {server_id} error: {e}")
                server_metrics['error'] = str(e)
                server_metrics['success'] = False
                return server_id, server_metrics, None

        # Query all servers in parallel
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.connections)) as executor:
            futures = [executor.submit(query_server_with_profiling, sid) for sid in self.connections]

            for future in concurrent.futures.as_completed(futures):
                try:
                    server_id, metrics, response = future.result()
                    per_server_metrics[server_id] = metrics
                    results[server_id] = response
                except Exception as e:
                    logger.error(f"Query error: {e}")

        profiling_data['client_metrics']['per_server_communication'] = per_server_metrics

        # 3. Collect server responses with detailed profiling
        successful_responses = {sid: r for sid, r in results.items()
                              if r and r.get('status') == 'success'}

        if len(successful_responses) >= 2:
            profiling_data['success'] = True

            # Store full server responses including detailed_profiling
            for server_id, response in successful_responses.items():
                profiling_data['server_responses'][server_id] = {
                    'timing': response.get('timing', {}),
                    'detailed_profiling': response.get('detailed_profiling', {}),
                    'server_id': response.get('server_id')
                }

                # Log server timing breakdown
                timing = response.get('timing', {})
                logger.info(f"\n  Server {server_id} timing breakdown:")
                logger.info(f"    Phase 1 (VDPF): {timing.get('phase1_time', 0):.2f} ms")
                logger.info(f"    Phase 2 (e/f):  {timing.get('phase2_time', 0):.2f} ms")
                logger.info(f"    Phase 3 (exchange): {timing.get('phase3_time', 0):.2f} ms")
                logger.info(f"    Phase 4 (reconstruct): {timing.get('phase4_time', 0):.2f} ms")
                logger.info(f"    Total: {timing.get('total', 0):.2f} ms")
        else:
            logger.error("Query failed: insufficient successful responses")
            profiling_data['success'] = False

        return profiling_data

    def _reconstruct_final_result(self, results):
        """Reconstruct final result"""
        # Get at least two server responses
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
            
            # Convert back to float
            if reconstructed > self.config.prime // 2:
                signed = reconstructed - self.config.prime
            else:
                signed = reconstructed
            
            reconstructed_vector[i] = signed / scale_factor
        
        return reconstructed_vector
    
    def _verify_result(self, node_id: int, reconstructed_vector: np.ndarray):
        """Verify correctness of reconstruction result"""
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
                logger.info(f"  Number of VDPF processes: {response.get('vdpf_processes')}")
                logger.info(f"  Data loaded: {response.get('data_loaded')}")
                logger.info(f"  Available triples: {response.get('triples_available')}")
            else:
                logger.error(f"Unable to get status from server {server_id}")

    def disconnect_from_servers(self):
        """Disconnect all server connections"""
        for server_id, sock in self.connections.items():
            try:
                sock.close()
                logger.info(f"Disconnected from server {server_id}")
            except:
                pass
        self.connections.clear()


def generate_markdown_report(dataset, query_details, avg_phases, avg_similarity):
    """Generate test report in Markdown format"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    markdown = f"""# Distributed Test Report - {dataset}

**Generated**: {timestamp}  
**Dataset**: {dataset}  
**Number of queries**: {len(query_details)}

## Detailed Query Results

| Query No. | Node ID | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruct) | Total Time | Cosine Similarity |
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

- **Phase 1 (VDPF Evaluation)**: {avg_phases['phase1']:.2f} seconds
- **Phase 2 (e/f Computation)**: {avg_phases['phase2']:.2f} seconds
- **Phase 3 (Data Exchange)**: {avg_phases['phase3']:.2f} seconds
- **Phase 4 (Reconstruction)**: {avg_phases['phase4']:.2f} seconds
- **Server Average Total**: {avg_phases['total']:.2f} seconds
- **Average Cosine Similarity**: {avg_similarity:.6f}

## Performance Analysis

### Time Distribution
"""

    # Calculate percentage of time for each phase
    total_avg = avg_phases['total']
    if total_avg > 0:
        phase1_pct = (avg_phases['phase1'] / total_avg) * 100
        phase2_pct = (avg_phases['phase2'] / total_avg) * 100
        phase3_pct = (avg_phases['phase3'] / total_avg) * 100
        phase4_pct = (avg_phases['phase4'] / total_avg) * 100

        markdown += f"""
- Phase 1 (VDPF Evaluation): {phase1_pct:.1f}%
- Phase 2 (e/f Computation): {phase2_pct:.1f}%
- Phase 3 (Data Exchange): {phase3_pct:.1f}%
- Phase 4 (Reconstruction): {phase4_pct:.1f}%

### Throughput
- Average query time: {total_avg:.2f} seconds
- Theoretical Throughput: {1/total_avg:.2f} queries/second
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
                        help='Do not save test reports')
    parser.add_argument('--config', type=str,
                        help='Server configuration file path')
    parser.add_argument('--status-only', action='store_true',
                        help='Only get server status')
    
    args = parser.parse_args()
    
    # Load customconfig
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
        # connectionServer
        if not client.connect_to_servers():
            logger.error("connectionServerfailed")
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

        logger.info(f"Will test queries on {len(random_nodes)} random nodes...\n")

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

        # Calculate averages
        if all_timings:
            logger.info(f"\n=== Average performance statistics ({len(all_timings)} successful queries) ===")
            avg_phases = {}
            for phase in ['phase1', 'phase2', 'phase3', 'phase4', 'total']:
                avg_phases[phase] = np.mean([t[phase] for t in all_timings])

            logger.info(f"  Phase 1 (VDPF Evaluation): {avg_phases['phase1']:.2f}seconds")
            logger.info(f"  Phase 2 (e/f Computation): {avg_phases['phase2']:.2f}seconds")
            logger.info(f"  Phase 3 (Data Exchange): {avg_phases['phase3']:.2f}seconds")
            logger.info(f"  Phase 4 (Reconstruction): {avg_phases['phase4']:.2f}seconds")
            logger.info(f"  Server average total: {avg_phases['total']:.2f}seconds")

            if all_similarities:
                avg_similarity = np.mean(all_similarities)
                logger.info(f"  Average Cosine similarity: {avg_similarity:.6f}")
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

                # Append mode, add separator
                with open(report_file, 'a', encoding='utf-8') as f:
                    # If file exists and not empty, add separator
                    f.seek(0, 2)  # Move to end of file
                    if f.tell() > 0:
                        f.write("\n\n---\n\n")
                    f.write(markdown_report)

                logger.info(f"\nTest report saved to: {report_file}")

    except KeyboardInterrupt:
        logger.info("\nUser interrupted")
    except Exception as e:
        logger.error(f"error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.disconnect_from_servers()


if __name__ == "__main__":
    main()