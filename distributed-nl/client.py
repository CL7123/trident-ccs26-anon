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

# Add project path
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
    # Default configuration - Client uses public IP
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
        
        # [CN]
        data_dir = f"~/trident/dataset/{dataset}"
        
        # [CN]
        neighbors_path = os.path.join(data_dir, "neighbors.bin")
        if os.path.exists(neighbors_path):
            # [CN]HNSW[CN]
            self.original_neighbors = self._load_hnsw_neighbors(neighbors_path)
            if self.original_neighbors is not None:
                # calculate[CN]number of nodes（[CN] / [CN]）
                num_layers = 3
                num_nodes = len(self.original_neighbors) // num_layers
                logger.info(f"[CN]HNSW[CN]: [CN]={len(self.original_neighbors)}, [CN]={num_nodes}")
            else:
                logger.warning("[CN]neighbors.bin[CN]")
        else:
            # [CN]ivecs[CN]groundtruth[CN]
            gt_path = os.path.join(data_dir, "gt.ivecs")
            if os.path.exists(gt_path):
                self.original_neighbors = self._load_ivecs(gt_path)
                logger.info(f"[CN]groundtruth[CN]: {self.original_neighbors.shape}")
            else:
                self.original_neighbors = None
                logger.warning("[CN]，[CN]")
        
        # [CN] - [CN]9001-9003
        self.servers_config = servers_config or {
            server_id: {
                "host": info["host"],
                "port": 9000 + server_id  # [CN]9001-9003[CN]
            }
            for server_id, info in SERVERS.items()
        }
        self.connections = {}
        self.connection_retry_count = 3
        self.connection_timeout = 10
        
    def _load_ivecs(self, filename):
        """[CN]ivecs[CN]"""
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
        """[CN]HNSW[CN]neighbors.bin[CN]"""
        try:
            import struct
            with open(filename, 'rb') as f:
                # [CN]header
                num_nodes = struct.unpack('<I', f.read(4))[0]
                num_layers = struct.unpack('<I', f.read(4))[0]
                max_neighbors = struct.unpack('<I', f.read(4))[0]
                _ = struct.unpack('<I', f.read(4))[0]  # [CN]0
                
                logger.info(f"HNSW[CN]: [CN]={num_nodes}, [CN]={num_layers}, [CN]={max_neighbors}")
                
                # [CN]：2[CN] + [CN]
                ints_per_node = 2 + num_layers * max_neighbors
                
                # create[CN]，[CN]
                # [CN] = node_id * num_layers + layer
                linear_neighbors = {}
                
                for node_id in range(num_nodes):
                    # [CN]
                    node_data = struct.unpack(f'<{ints_per_node}I', f.read(ints_per_node * 4))
                    
                    # [CN]2[CN]
                    # [CN]：[metadata1, metadata2, layer0_neighbors, layer1_neighbors, layer2_neighbors]
                    
                    # [CN]
                    for layer in range(num_layers):
                        # [CN]2[CN]
                        start_idx = 2 + layer * max_neighbors
                        end_idx = start_idx + max_neighbors
                        layer_neighbors = list(node_data[start_idx:end_idx])
                        
                        # calculate[CN]
                        linear_idx = node_id * num_layers + layer
                        
                        # [CN]128[CN]（[CN]4294967295[CN]）
                        linear_neighbors[linear_idx] = layer_neighbors
                
                return linear_neighbors
                
        except Exception as e:
            logger.error(f"[CN]HNSW[CN]: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def connect_to_servers(self):
        """connect[CN]"""
        successful_connections = 0
        
        for server_id, server_info in self.servers_config.items():
            host = server_info["host"]
            port = server_info["port"]
            
            connected = False
            for attempt in range(self.connection_retry_count):
                try:
                    logger.info(f"[CN]connect[CN] {server_id} ({host}:{port})，[CN] {attempt + 1} [CN]...")
                    
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(self.connection_timeout)
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    
                    sock.connect((host, port))
                    self.connections[server_id] = sock
                    logger.info(f"[CN]connect[CN] {server_id}")
                    successful_connections += 1
                    connected = True
                    break
                    
                except socket.timeout:
                    logger.warning(f"connect[CN] {server_id} [CN]")
                except ConnectionRefusedError:
                    logger.warning(f"[CN] {server_id} [CN]connect")
                except Exception as e:
                    logger.warning(f"connect[CN] {server_id} [CN]: {e}")
                
                if attempt < self.connection_retry_count - 1:
                    time.sleep(2)  # [CN]
            
            if not connected:
                logger.error(f"[CN]connect[CN] {server_id}")
        
        logger.info(f"[CN]connect[CN] {successful_connections}/{len(self.servers_config)} [CN]")
        return successful_connections >= 2  # [CN]2servers
    
    def _send_request(self, server_id: int, request: dict) -> Optional[dict]:
        """[CN]send[CN]，[CN]process"""
        if server_id not in self.connections:
            logger.error(f"[CN]connect[CN] {server_id}")
            return None
        
        sock = self.connections[server_id]
        
        try:
            # [CN]send[CN]
            if 'dpf_key' in request:
                # [CN]，[CN]
                old_timeout = sock.gettimeout()
                sock.settimeout(60)  # 60[CN]
                
                # [CN]send[CN]
                BinaryProtocol.send_binary_request(
                    sock, 
                    request['command'],
                    request['dpf_key'],
                    request.get('query_id')
                )
                
                # receive[CN]
                response = BinaryProtocol.receive_response(sock)
                
                # [CN]
                sock.settimeout(old_timeout)
                return response
            else:
                # [CN]JSON
                request_data = json.dumps(request).encode()
                sock.sendall(len(request_data).to_bytes(4, 'big'))
                sock.sendall(request_data)
                
                # receive[CN]
                length_bytes = sock.recv(4)
                if not length_bytes:
                    raise ConnectionError("connect[CN]")
                
                length = int.from_bytes(length_bytes, 'big')
                data = b''
                while len(data) < length:
                    chunk = sock.recv(min(length - len(data), 4096))
                    if not chunk:
                        raise ConnectionError("receive[CN]connect[CN]")
                    data += chunk
                
                return json.loads(data.decode())
                
        except Exception as e:
            logger.error(f"[CN] {server_id} [CN]: {e}")
            return None
    
    def test_distributed_neighbor_query(self, query_node_id: int = 0):
        """[CN]"""
        # [CN]VDPF[CN] - [CN]
        keys = self.dpf_wrapper.generate_keys('neighbor', query_node_id)
        
        # [CN]ID
        query_id = f'nl_distributed_test_{time.time()}_{query_node_id}'
        
        logger.info(f"[CN]，[CN]ID: {query_node_id}, [CN]ID: {query_id}")
        
        # [CN]
        start_time = time.time()
        
        def query_server(server_id):
            request = {
                'command': 'query_neighbor_list',
                'dpf_key': keys[server_id - 1],
                'query_id': query_id
            }
            response = self._send_request(server_id, request)
            return server_id, response
        
        # [CN]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.connections)) as executor:
            futures = [executor.submit(query_server, sid) for sid in self.connections]
            results = {}
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    server_id, response = future.result()
                    results[server_id] = response
                except Exception as e:
                    logger.error(f"[CN]: {e}")
        
        # [CN]
        successful_responses = {sid: r for sid, r in results.items() 
                              if r and r.get('status') == 'success'}
        
        if len(successful_responses) < 2:
            logger.error("[CN]：[CN]2[CN]")
            for server_id, result in results.items():
                if not result or result.get('status') != 'success':
                    logger.error(f"[CN] {server_id}: {result}")
            return None, None
        
        # [CN]
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
        
        # calculate[CN]
        avg_timings = {}
        for phase in ['phase1', 'phase2', 'phase3', 'phase4', 'total']:
            phase_times = [t[phase] for t in timings.values()]
            avg_timings[phase] = np.mean(phase_times) if phase_times else 0
        
        # [CN]
        final_result = self._reconstruct_neighbor_list(successful_responses)
        
        # [CN]
        accuracy = self._verify_neighbor_result(query_node_id, final_result)
        
        # print[CN]
        total_time = time.time() - start_time
        logger.info(f"\n[CN]:")
        logger.info(f"  [CN]: {total_time:.2f}[CN]")
        logger.info(f"  [CN]1 (VDPF[CN]): {avg_timings['phase1']:.2f}[CN]")
        logger.info(f"  [CN]2 (e/fcalculate): {avg_timings['phase2']:.2f}[CN]")
        logger.info(f"  [CN]3 ([CN]): {avg_timings['phase3']:.2f}[CN]")
        logger.info(f"  [CN]4 ([CN]): {avg_timings['phase4']:.2f}[CN]")
        logger.info(f"  [CN]: {avg_timings['total']:.2f}[CN]")
        if accuracy is not None:
            logger.info(f"  [CN]: {accuracy:.2%}")
        logger.info(f"  return[CN]: {len(final_result) if final_result is not None else 0}")
        
        return avg_timings, final_result
    
    def _reconstruct_neighbor_list(self, results):
        """[CN]"""
        # [CN]servers[CN]
        server_ids = sorted([sid for sid, r in results.items() 
                           if r and r.get('status') == 'success'])[:2]
        
        if len(server_ids) < 2:
            logger.error("[CN]：[CN]2[CN]")
            return None
        
        # [CN]
        first_result = results[server_ids[0]]['result_share']
        k_neighbors = len(first_result)
        
        # [CN]
        reconstructed_neighbors = []
        
        for i in range(k_neighbors):
            shares = [
                Share(results[server_ids[0]]['result_share'][i], server_ids[0]),
                Share(results[server_ids[1]]['result_share'][i], server_ids[1])
            ]
            
            reconstructed = self.mpc.reconstruct(shares)
            
            # process[CN]
            # [CN] field_size-1 ([CN] prime-1) [CN] -1
            # [CN]HNSW[CN] 4294967295 [CN]
            if reconstructed == self.config.prime - 1:
                # [CN] -1，[CN]HNSW[CN] 4294967295
                neighbor_idx = 4294967295
            elif reconstructed >= self.config.num_docs:
                # [CN]
                neighbor_idx = 4294967295
            else:
                # [CN]
                neighbor_idx = reconstructed
            
            reconstructed_neighbors.append(int(neighbor_idx))
        
        return reconstructed_neighbors
    
    def _verify_neighbor_result(self, query_node_id: int, reconstructed_neighbors: List[int]):
        """[CN]"""
        try:
            if self.original_neighbors is None or reconstructed_neighbors is None:
                return None
            
            # [CN]original_neighbors[CN]
            if query_node_id not in self.original_neighbors:
                logger.warning(f"[CN] {query_node_id} [CN]")
                return None
            
            # [CN]（[CN]）
            original_layer_neighbors = self.original_neighbors[query_node_id]
            
            # calculate[CN]ID[CN]（[CN]）
            num_layers = 3  # HNSW[CN]
            actual_node_id = query_node_id // num_layers
            layer = query_node_id % num_layers
            
            # [CN]share_data.py[CN]bug，[CN]
            # [CN]2[CN]
            # [CN]2[CN]
            
            # [CN]，[CN]
            # [CN]：[CN]，[CN]
            
            # [CN]（[CN]4294967295[CN]）
            valid_original = [n for n in original_layer_neighbors if n != 4294967295]
            valid_reconstructed = [n for n in reconstructed_neighbors if n != 4294967295 and n < self.config.num_docs]
            
            # calculate[CN] - [CN]
            if len(valid_original) == 0:
                # [CN]，[CN]
                accuracy = 1.0 if len(valid_reconstructed) == 0 else 0.0
            else:
                # calculate[CN]
                # [CN]，[CN]
                matches = len(set(valid_original) & set(valid_reconstructed))
                # [CN]，[CN]
                if len(valid_reconstructed) != len(valid_original):
                    # [CN]（[CN]）
                    accuracy = matches / max(len(valid_original), len(valid_reconstructed))
                else:
                    accuracy = matches / len(valid_original)
            
            logger.info(f"[CN] ([CN]={query_node_id}, [CN]={actual_node_id}, [CN]={layer}):")
            logger.info(f"  [CN]10[CN]: {original_layer_neighbors[:10]}")
            logger.info(f"  [CN]10[CN]: {original_layer_neighbors[-10:]}")
            logger.info(f"  [CN]10[CN]: {reconstructed_neighbors[:10]}")
            logger.info(f"  [CN]10[CN]: {reconstructed_neighbors[-10:]}")
            logger.info(f"  [CN]: {len(valid_original)} [CN]")
            logger.info(f"  [CN]: {len(valid_reconstructed)} [CN]")
            if len(valid_original) > 0:
                logger.info(f"  [CN]: {matches}/{len(valid_original)}")
            
            return accuracy
                
        except Exception as e:
            logger.error(f"[CN]: {e}")
            return None
    
    def get_server_status(self):
        """[CN]"""
        logger.info("[CN]...")
        
        for server_id in self.connections:
            request = {'command': 'get_status'}
            response = self._send_request(server_id, request)
            
            if response and response.get('status') == 'success':
                logger.info(f"\n[CN] {server_id} [CN]:")
                logger.info(f"  [CN]: {response.get('mode')}")
                logger.info(f"  [CN]: {response.get('host')}:{response.get('port')}")
                logger.info(f"  Dataset: {response.get('dataset')}")
                logger.info(f"  VDPF[CN]: {response.get('vdpf_processes')}")
                logger.info(f"  [CN]: {response.get('data_loaded')}")
                logger.info(f"  [CN]: {response.get('triples_available')}")
            else:
                logger.error(f"[CN] {server_id} [CN]")
    
    def disconnect_from_servers(self):
        """[CN]connect"""
        for server_id, sock in self.connections.items():
            try:
                sock.close()
                logger.info(f"[CN] {server_id} [CN]connect")
            except:
                pass
        self.connections.clear()


def generate_markdown_report(dataset, query_details, avg_phases, avg_accuracy):
    """[CN]Markdown[CN]"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    markdown = f"""# [CN] - {dataset}

**[CN]**: {timestamp}  
**Dataset**: {dataset}  
**[CN]**: {len(query_details)}

## [CN]

| [CN] | [CN]ID | [CN]1 (VDPF) | [CN]2 (e/f) | [CN]3 ([CN]) | [CN]4 ([CN]) | [CN] | [CN] |
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
## [CN]

- **[CN]1 (VDPF[CN])**: {avg_phases['phase1']:.2f}[CN]
- **[CN]2 (e/fcalculate)**: {avg_phases['phase2']:.2f}[CN]
- **[CN]3 ([CN])**: {avg_phases['phase3']:.2f}[CN]
- **[CN]4 ([CN])**: {avg_phases['phase4']:.2f}[CN]
- **[CN]**: {avg_phases['total']:.2f}[CN]
- **[CN]**: {avg_accuracy:.2%}

## [CN]

### [CN]
"""
    
    # calculate[CN]
    total_avg = avg_phases['total']
    if total_avg > 0:
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
    parser = argparse.ArgumentParser(description='Distributed neighbor list query client')
    parser.add_argument('--dataset', type=str, default='siftsmall', 
                        choices=['siftsmall', 'laion', 'tripclick', 'ms_marco', 'nfcorpus'],
                        help='Dataset[CN] ([CN]: siftsmall)')
    parser.add_argument('--num-queries', type=int, default=10,
                        help='[CN]Number of queries[CN] ([CN]: 10)')
    parser.add_argument('--no-report', action='store_true',
                        help='[CN]')
    parser.add_argument('--config', type=str,
                        help='[CN]')
    parser.add_argument('--status-only', action='store_true',
                        help='[CN]')
    
    args = parser.parse_args()
    
    # [CN]
    servers_config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                servers_config = json.load(f)
        except Exception as e:
            logger.error(f"Load configuration[CN]: {e}")
            return
    
    logger.info(f"=== [CN] - Dataset: {args.dataset} ===")
    
    client = DistributedNeighborClient(args.dataset, servers_config)
    
    try:
        # connect[CN]
        if not client.connect_to_servers():
            logger.error("connect[CN]")
            return
        
        # [CN]
        if args.status_only:
            client.get_server_status()
            return
        
        all_timings = []
        all_accuracies = []
        query_details = []
        
        # [CN]（[CN]）
        total_nodes = len(client.original_neighbors) if client.original_neighbors is not None else 1000
        
        # [CN]
        random_nodes = random.sample(range(total_nodes), min(args.num_queries, total_nodes))
        
        logger.info(f"[CN] {len(random_nodes)} [CN]...\n")
        
        for i, node_id in enumerate(random_nodes):
            logger.info(f"[CN] {i+1}/{len(random_nodes)}: [CN] {node_id} [CN]")
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
                    'neighbors': neighbors[:10] if neighbors else []  # [CN]10[CN]
                })
        
        # calculate[CN]
        if all_timings:
            logger.info(f"\n=== [CN] ({len(all_timings)} [CN]) ===")
            avg_phases = {}
            for phase in ['phase1', 'phase2', 'phase3', 'phase4', 'total']:
                avg_phases[phase] = np.mean([t[phase] for t in all_timings])
            
            logger.info(f"  [CN]1 (VDPF[CN]): {avg_phases['phase1']:.2f}[CN]")
            logger.info(f"  [CN]2 (e/fcalculate): {avg_phases['phase2']:.2f}[CN]")
            logger.info(f"  [CN]3 ([CN]): {avg_phases['phase3']:.2f}[CN]")
            logger.info(f"  [CN]4 ([CN]): {avg_phases['phase4']:.2f}[CN]")
            logger.info(f"  [CN]: {avg_phases['total']:.2f}[CN]")
            
            if all_accuracies:
                avg_accuracy = np.mean(all_accuracies)
                logger.info(f"  [CN]: {avg_accuracy:.2%}")
            else:
                avg_accuracy = 0.0
            
            # [CN]
            if not args.no_report and query_details:
                report_file = "~/trident/distributed-nl/nl_result.md"
                markdown_report = generate_markdown_report(
                    args.dataset, 
                    query_details, 
                    avg_phases,
                    avg_accuracy
                )
                
                # [CN]，[CN]
                with open(report_file, 'a', encoding='utf-8') as f:
                    # [CN]，[CN]
                    f.seek(0, 2)  # [CN]
                    if f.tell() > 0:
                        f.write("\n\n---\n\n")
                    f.write(markdown_report)
                
                logger.info(f"\n[CN]: {report_file}")
            
    except KeyboardInterrupt:
        logger.info("\n[CN]")
    except Exception as e:
        logger.error(f"[CN]: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.disconnect_from_servers()


if __name__ == "__main__":
    main()