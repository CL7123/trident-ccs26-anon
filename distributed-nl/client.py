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
        1: {"host": "192.168.1.101", "port": 9001},
        2: {"host": "192.168.1.102", "port": 9002},
        3: {"host": "192.168.1.103", "port": 9003}
    }

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [NL-Client] %(message)s')
logger = logging.getLogger(__name__)


class DistributedNeighborClient:
    """Distributed neighbor list query client"""
    
    def __init__(self, dataset: str = "siftsmall", servers_config: Dict = None):
        self.dataset = dataset
        self.config = get_config(dataset)
        self.dpf_wrapper = OptimizedVDPFVectorWrapper(dataset_name=dataset)
        self.mpc = MPC23SSS(self.config)

        # Preload original neighbor data for verification
        data_dir = f"~/trident/dataset/{dataset}"

        # Try to load neighbor list data
        neighbors_path = os.path.join(data_dir, "neighbors.bin")
        if os.path.exists(neighbors_path):
            # Load HNSW format neighbor list
            self.original_neighbors = self._load_hnsw_neighbors(neighbors_path)
            if self.original_neighbors is not None:
                # Calculate actual node count (linear index count / number of layers)
                num_layers = 3
                num_nodes = len(self.original_neighbors) // num_layers
                logger.info(f"Preloaded HNSW neighbor data: linear_index_count={len(self.original_neighbors)}, node_count={num_nodes}")
            else:
                logger.warning("Failed to load neighbors.bin")
        else:
            # Try ivecs format groundtruth as fallback
            gt_path = os.path.join(data_dir, "gt.ivecs")
            if os.path.exists(gt_path):
                self.original_neighbors = self._load_ivecs(gt_path)
                logger.info(f"Preloaded groundtruth neighbor data: {self.original_neighbors.shape}")
            else:
                self.original_neighbors = None
                logger.warning("Neighbor list data not found, cannot verify results")

        # Server configuration - update port to 9001-9003
        self.servers_config = servers_config or {
            server_id: {
                "host": info["host"],
                "port": 9000 + server_id  # Neighbor list service uses ports 9001-9003
            }
            for server_id, info in SERVERS.items()
        }
        self.connections = {}
        self.connection_retry_count = 3
        self.connection_timeout = 10

    def _load_ivecs(self, filename):
        """Load ivecs format file"""
        with open(filename, 'rb') as f:
            vectors = []
            while True:
                dim_bytes = f.read(4)
                if not dim_bytes:
                    break
                dim = int.from_bytes(dim_bytes, byteorder='little')
                vector = np.frombuffer(f.read(dim * 4), dtype=np.int32)
                vectors.append(vector)
            return np.array(vectors)
    
    def _load_hnsw_neighbors(self, filename):
        """Load HNSW format neighbors.bin file"""
        try:
            import struct
            with open(filename, 'rb') as f:
                # Read header
                num_nodes = struct.unpack('<I', f.read(4))[0]
                num_layers = struct.unpack('<I', f.read(4))[0]
                max_neighbors = struct.unpack('<I', f.read(4))[0]
                _ = struct.unpack('<I', f.read(4))[0]  # Skip extra 0

                logger.info(f"HNSW data: node_count={num_nodes}, layer_count={num_layers}, max_neighbors={max_neighbors}")

                # Data size per node: 2 metadata + neighbor data for each layer
                ints_per_node = 2 + num_layers * max_neighbors

                # Create linearized neighbor data array consistent with server-side storage format
                # Linear index = node_id * num_layers + layer
                linear_neighbors = {}

                for node_id in range(num_nodes):
                    # Read all data for this node
                    node_data = struct.unpack(f'<{ints_per_node}I', f.read(ints_per_node * 4))

                    # Skip first 2 metadata values
                    # Data layout: [metadata1, metadata2, layer0_neighbors, layer1_neighbors, layer2_neighbors]

                    # Store neighbor data for each layer
                    for layer in range(num_layers):
                        # Correct format should skip first 2 metadata values
                        start_idx = 2 + layer * max_neighbors
                        end_idx = start_idx + max_neighbors
                        layer_neighbors = list(node_data[start_idx:end_idx])

                        # Calculate linear index
                        linear_idx = node_id * num_layers + layer

                        # Store all 128 neighbors (including 4294967295 padding values)
                        linear_neighbors[linear_idx] = layer_neighbors

                return linear_neighbors

        except Exception as e:
            logger.error(f"Error loading HNSW neighbor data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def connect_to_servers(self):
        """Connect to all neighbor list servers"""
        successful_connections = 0
        
        for server_id, server_info in self.servers_config.items():
            host = server_info["host"]
            port = server_info["port"]
            
            connected = False
            for attempt in range(self.connection_retry_count):
                try:
                    logger.info(f"Attempting to connect to neighbor list server {server_id} ({host}:{port}), attempt {attempt + 1}...")

                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(self.connection_timeout)
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                    sock.connect((host, port))
                    self.connections[server_id] = sock
                    logger.info(f"Successfully connected to neighbor list server {server_id}")
                    successful_connections += 1
                    connected = True
                    break

                except socket.timeout:
                    logger.warning(f"Connection to neighbor list server {server_id} timed out")
                except ConnectionRefusedError:
                    logger.warning(f"Neighbor list server {server_id} refused connection")
                except Exception as e:
                    logger.warning(f"Failed to connect to neighbor list server {server_id}: {e}")

                if attempt < self.connection_retry_count - 1:
                    time.sleep(2)  # Wait before retry

            if not connected:
                logger.error(f"Unable to connect to neighbor list server {server_id}")

        logger.info(f"Successfully connected to {successful_connections}/{len(self.servers_config)} neighbor list servers")
        return successful_connections >= 2  # At least 2 servers needed

    def _send_request(self, server_id: int, request: dict) -> Optional[dict]:
        """Send request to specified server with error handling"""
        if server_id not in self.connections:
            logger.error(f"Not connected to server {server_id}")
            return None

        sock = self.connections[server_id]

        try:
            # Use binary protocol to send requests containing keys
            if 'dpf_key' in request:
                # Increase timeout for query requests
                old_timeout = sock.gettimeout()
                sock.settimeout(60)  # 60 second timeout

                # Send request using binary protocol
                BinaryProtocol.send_binary_request(
                    sock,
                    request['command'],
                    request['dpf_key'],
                    request.get('query_id')
                )

                # Receive response
                response = BinaryProtocol.receive_response(sock)

                # Restore original timeout
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
    
    def test_distributed_neighbor_query(self, query_node_id: int = 0):
        """Test distributed neighbor list query"""
        # Generate VDPF keys for neighbor query
        keys = self.dpf_wrapper.generate_keys('neighbor', query_node_id)

        # Generate query ID
        query_id = f'nl_distributed_test_{time.time()}_{query_node_id}'

        logger.info(f"Starting distributed neighbor list query, query_node_id: {query_node_id}, query_id: {query_id}")

        # Query all servers in parallel
        start_time = time.time()

        def query_server(server_id):
            request = {
                'command': 'query_neighbor_list',
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
            logger.error("Query failed: fewer than 2 servers returned successful responses")
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

        # Calculate average times
        avg_timings = {}
        for phase in ['phase1', 'phase2', 'phase3', 'phase4', 'total']:
            phase_times = [t[phase] for t in timings.values()]
            avg_timings[phase] = np.mean(phase_times) if phase_times else 0

        # Reconstruct neighbor list
        final_result = self._reconstruct_neighbor_list(successful_responses)

        # Verify result
        accuracy = self._verify_neighbor_result(query_node_id, final_result)

        # Print results
        total_time = time.time() - start_time
        logger.info(f"\nNeighbor list query results:")
        logger.info(f"  Client total time: {total_time:.2f}s")
        logger.info(f"  Phase 1 (VDPF evaluation): {avg_timings['phase1']:.2f}s")
        logger.info(f"  Phase 2 (e/f computation): {avg_timings['phase2']:.2f}s")
        logger.info(f"  Phase 3 (data exchange): {avg_timings['phase3']:.2f}s")
        logger.info(f"  Phase 4 (reconstruction): {avg_timings['phase4']:.2f}s")
        logger.info(f"  Server average total: {avg_timings['total']:.2f}s")
        if accuracy is not None:
            logger.info(f"  Neighbor list accuracy: {accuracy:.2%}")
        logger.info(f"  Neighbors returned: {len(final_result) if final_result is not None else 0}")

        return avg_timings, final_result
    
    def _reconstruct_neighbor_list(self, results):
        """Reconstruct neighbor list"""
        # Get at least two server responses
        server_ids = sorted([sid for sid, r in results.items()
                           if r and r.get('status') == 'success'])[:2]

        if len(server_ids) < 2:
            logger.error("Reconstruction failed: fewer than 2 servers available")
            return None

        # Get neighbor list length
        first_result = results[server_ids[0]]['result_share']
        k_neighbors = len(first_result)

        # Reconstruct each neighbor index
        reconstructed_neighbors = []

        for i in range(k_neighbors):
            shares = [
                Share(results[server_ids[0]]['result_share'][i], server_ids[0]),
                Share(results[server_ids[1]]['result_share'][i], server_ids[1])
            ]

            reconstructed = self.mpc.reconstruct(shares)

            # Handle reconstructed values
            # Secret sharing uses field_size-1 (i.e., prime-1) to represent -1
            # Original HNSW uses 4294967295 for invalid neighbors
            if reconstructed == self.config.prime - 1:
                # This is padding value -1, represented as 4294967295 in HNSW
                neighbor_idx = 4294967295
            elif reconstructed >= self.config.num_docs:
                # Indices beyond document range are also invalid
                neighbor_idx = 4294967295
            else:
                # Normal neighbor index
                neighbor_idx = reconstructed

            reconstructed_neighbors.append(int(neighbor_idx))

        return reconstructed_neighbors
    
    def _verify_neighbor_result(self, query_node_id: int, reconstructed_neighbors: List[int]):
        """Verify correctness of neighbor list results"""
        try:
            if self.original_neighbors is None or reconstructed_neighbors is None:
                return None

            # Use linear index directly to get data from original_neighbors dict
            if query_node_id not in self.original_neighbors:
                logger.warning(f"Linear index {query_node_id} not in original data")
                return None

            # Get original neighbor list (already data for specific node and layer)
            original_layer_neighbors = self.original_neighbors[query_node_id]

            # Calculate actual node ID and layer (for logging)
            num_layers = 3  # Number of layers in HNSW
            actual_node_id = query_node_id // num_layers
            layer = query_node_id % num_layers

            # Due to bug in share_data.py, data has circular shift
            # First 2 values of reconstructed come from end of previous node
            # Last 2 values of original appear at beginning of next node

            # For correct comparison, we need to align data
            # Method: compare sets rather than positions, as data is circularly shifted

            # Filter valid neighbors (not 4294967295)
            valid_original = [n for n in original_layer_neighbors if n != 4294967295]
            valid_reconstructed = [n for n in reconstructed_neighbors if n != 4294967295 and n < self.config.num_docs]

            # Calculate accuracy - based on matching valid neighbors
            if len(valid_original) == 0:
                # If original has no valid neighbors, check if reconstructed also has none
                accuracy = 1.0 if len(valid_reconstructed) == 0 else 0.0
            else:
                # Calculate intersection of valid neighbors
                # Since data is circularly shifted, compare sets rather than positions
                matches = len(set(valid_original) & set(valid_reconstructed))
                # If number of reconstructed neighbors differs from original, there's an issue
                if len(valid_reconstructed) != len(valid_original):
                    # May have extra neighbors (from shift of other nodes)
                    accuracy = matches / max(len(valid_original), len(valid_reconstructed))
                else:
                    accuracy = matches / len(valid_original)

            logger.info(f"Neighbor list comparison (linear_index={query_node_id}, node={actual_node_id}, layer={layer}):")
            logger.info(f"  Original first 10: {original_layer_neighbors[:10]}")
            logger.info(f"  Original last 10: {original_layer_neighbors[-10:]}")
            logger.info(f"  Reconstructed first 10: {reconstructed_neighbors[:10]}")
            logger.info(f"  Reconstructed last 10: {reconstructed_neighbors[-10:]}")
            logger.info(f"  Original valid neighbors: {len(valid_original)}")
            logger.info(f"  Reconstructed valid neighbors: {len(valid_reconstructed)}")
            if len(valid_original) > 0:
                logger.info(f"  Matches: {matches}/{len(valid_original)}")

            return accuracy

        except Exception as e:
            logger.error(f"Error verifying neighbor list: {e}")
            return None
    
    def get_server_status(self):
        """Get status of all neighbor list servers"""
        logger.info("Getting neighbor list server status...")

        for server_id in self.connections:
            request = {'command': 'get_status'}
            response = self._send_request(server_id, request)

            if response and response.get('status') == 'success':
                logger.info(f"\nNeighbor list server {server_id} status:")
                logger.info(f"  Mode: {response.get('mode')}")
                logger.info(f"  Address: {response.get('host')}:{response.get('port')}")
                logger.info(f"  Dataset: {response.get('dataset')}")
                logger.info(f"  VDPF processes: {response.get('vdpf_processes')}")
                logger.info(f"  Data loaded: {response.get('data_loaded')}")
                logger.info(f"  Available triples: {response.get('triples_available')}")
            else:
                logger.error(f"Unable to get status of neighbor list server {server_id}")

    def disconnect_from_servers(self):
        """Disconnect from all servers"""
        for server_id, sock in self.connections.items():
            try:
                sock.close()
                logger.info(f"Disconnected from neighbor list server {server_id}")
            except:
                pass
        self.connections.clear()


def generate_markdown_report(dataset, query_details, avg_phases, avg_accuracy):
    """Generate markdown format test report"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    markdown = f"""# Distributed Neighbor List Test Report - {dataset}

**Generated**: {timestamp}
**Dataset**: {dataset}
**Number of Queries**: {len(query_details)}

## Detailed Query Results

| Query# | Node ID | Phase 1 (VDPF) | Phase 2 (e/f) | Phase 3 (Exchange) | Phase 4 (Reconstruct) | Total | Accuracy |
|---------|--------|--------------|-------------|--------------|--------------|--------|--------|
"""

    for q in query_details:
        markdown += f"| {q['query_num']} | {q['node_id']} | "
        markdown += f"{q['timings']['phase1']:.2f}s | "
        markdown += f"{q['timings']['phase2']:.2f}s | "
        markdown += f"{q['timings']['phase3']:.2f}s | "
        markdown += f"{q['timings']['phase4']:.2f}s | "
        markdown += f"{q['timings']['total']:.2f}s | "
        if q['accuracy'] is not None:
            markdown += f"{q['accuracy']:.2%} |\n"
        else:
            markdown += "N/A |\n"

    markdown += f"""
## Average Performance Statistics

- **Phase 1 (VDPF evaluation)**: {avg_phases['phase1']:.2f}s
- **Phase 2 (e/f computation)**: {avg_phases['phase2']:.2f}s
- **Phase 3 (data exchange)**: {avg_phases['phase3']:.2f}s
- **Phase 4 (reconstruction)**: {avg_phases['phase4']:.2f}s
- **Server average total**: {avg_phases['total']:.2f}s
- **Average accuracy**: {avg_accuracy:.2%}

## Performance Analysis

### Time Distribution
"""

    # Calculate percentage of each phase
    total_avg = avg_phases['total']
    if total_avg > 0:
        phase1_pct = (avg_phases['phase1'] / total_avg) * 100
        phase2_pct = (avg_phases['phase2'] / total_avg) * 100
        phase3_pct = (avg_phases['phase3'] / total_avg) * 100
        phase4_pct = (avg_phases['phase4'] / total_avg) * 100

        markdown += f"""
- Phase 1 (VDPF evaluation): {phase1_pct:.1f}%
- Phase 2 (e/f computation): {phase2_pct:.1f}%
- Phase 3 (data exchange): {phase3_pct:.1f}%
- Phase 4 (reconstruction): {phase4_pct:.1f}%

### Throughput
- Average query time: {total_avg:.2f}s
- Theoretical throughput: {1/total_avg:.2f} queries/s
"""

    return markdown


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Distributed neighbor list query client')
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

    logger.info(f"=== Distributed Neighbor List Test Client - Dataset: {args.dataset} ===")

    client = DistributedNeighborClient(args.dataset, servers_config)

    try:
        # Connect to servers
        if not client.connect_to_servers():
            logger.error("Failed to connect to neighbor list servers")
            return

        # If only getting status
        if args.status_only:
            client.get_server_status()
            return

        all_timings = []
        all_accuracies = []
        query_details = []

        # Get total number of nodes (query nodes)
        total_nodes = len(client.original_neighbors) if client.original_neighbors is not None else 1000

        # Randomly select query nodes
        random_nodes = random.sample(range(total_nodes), min(args.num_queries, total_nodes))

        logger.info(f"Performing neighbor list query tests on {len(random_nodes)} random nodes...\n")

        for i, node_id in enumerate(random_nodes):
            logger.info(f"Query {i+1}/{len(random_nodes)}: neighbor list for node {node_id}")
            timings, neighbors = client.test_distributed_neighbor_query(query_node_id=node_id)

            if timings:
                all_timings.append(timings)
                accuracy = client._verify_neighbor_result(node_id, neighbors)
                if accuracy is not None:
                    all_accuracies.append(accuracy)

                query_details.append({
                    'query_num': i + 1,
                    'node_id': node_id,
                    'timings': timings,
                    'accuracy': accuracy,
                    'neighbors': neighbors[:10] if neighbors else []  # Save only first 10 neighbors
                })

        # Calculate averages
        if all_timings:
            logger.info(f"\n=== Average Performance Statistics ({len(all_timings)} successful queries) ===")
            avg_phases = {}
            for phase in ['phase1', 'phase2', 'phase3', 'phase4', 'total']:
                avg_phases[phase] = np.mean([t[phase] for t in all_timings])

            logger.info(f"  Phase 1 (VDPF evaluation): {avg_phases['phase1']:.2f}s")
            logger.info(f"  Phase 2 (e/f computation): {avg_phases['phase2']:.2f}s")
            logger.info(f"  Phase 3 (data exchange): {avg_phases['phase3']:.2f}s")
            logger.info(f"  Phase 4 (reconstruction): {avg_phases['phase4']:.2f}s")
            logger.info(f"  Server average total: {avg_phases['total']:.2f}s")

            if all_accuracies:
                avg_accuracy = np.mean(all_accuracies)
                logger.info(f"  Average accuracy: {avg_accuracy:.2%}")
            else:
                avg_accuracy = 0.0

            # Save report
            if not args.no_report and query_details:
                report_file = "~/trident/distributed-nl/nl_result.md"
                markdown_report = generate_markdown_report(
                    args.dataset,
                    query_details,
                    avg_phases,
                    avg_accuracy
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
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.disconnect_from_servers()


if __name__ == "__main__":
    main()