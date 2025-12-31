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
    from config import SERVERS
except ImportError:
    # Default configuration
    SERVERS = {
        1: {"host": "192.168.1.101", "port": 8001},
        2: {"host": "192.168.1.102", "port": 8002},
        3: {"host": "192.168.1.103", "port": 8003}
    }

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [Client] %(message)s')
logger = logging.getLogger(__name__)


class DistributedClient:
    """[CN]"""
    
    def __init__(self, dataset: str = "siftsmall", servers_config: Dict = None):
        self.dataset = dataset
        self.config = get_config(dataset)
        self.dpf_wrapper = OptimizedVDPFVectorWrapper(dataset_name=dataset)
        self.mpc = MPC23SSS(self.config)
        
        # [CN]
        data_dir = f"~/trident/dataset/{dataset}"
        loader = DatasetLoader(data_dir)
        self.original_nodes = loader.load_nodes()
        logger.info(f"[CN] {len(self.original_nodes)} [CN]")
        
        # [CN]
        self.servers_config = servers_config or SERVERS
        self.connections = {}
        self.connection_retry_count = 3
        self.connection_timeout = 10
        
    def connect_to_servers(self):
        """connect[CN]，[CN]"""
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
        
        logger.info(f"[CN]connect[CN] {successful_connections}/{len(self.servers_config)} servers")
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
    
    def test_distributed_query(self, node_id: int = 1723):
        """[CN]"""
        # [CN]VDPF[CN]
        keys = self.dpf_wrapper.generate_keys('node', node_id)
        
        # [CN]ID
        query_id = f'distributed_test_{time.time()}_{node_id}'
        
        logger.info(f"[CN]，[CN]ID: {node_id}, [CN]ID: {query_id}")
        
        # [CN]
        start_time = time.time()
        
        def query_server(server_id):
            request = {
                'command': 'query_node_vector',
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
        final_result = self._reconstruct_final_result(successful_responses)
        
        # [CN]
        similarity = self._verify_result(node_id, final_result)
        
        # print[CN]
        total_time = time.time() - start_time
        logger.info(f"\n[CN]:")
        logger.info(f"  [CN]: {total_time:.2f}[CN]")
        logger.info(f"  [CN]1 (VDPF[CN]): {avg_timings['phase1']:.2f}[CN]")
        logger.info(f"  [CN]2 (e/fcalculate): {avg_timings['phase2']:.2f}[CN]")
        logger.info(f"  [CN]3 ([CN]): {avg_timings['phase3']:.2f}[CN]")
        logger.info(f"  [CN]4 ([CN]): {avg_timings['phase4']:.2f}[CN]")
        logger.info(f"  [CN]: {avg_timings['total']:.2f}[CN]")
        if similarity is not None:
            logger.info(f"  [CN]: {similarity:.6f}")
        
        return avg_timings, final_result
    
    def _reconstruct_final_result(self, results):
        """[CN]"""
        # [CN]servers[CN]
        server_ids = sorted([sid for sid, r in results.items() 
                           if r and r.get('status') == 'success'])[:2]
        
        if len(server_ids) < 2:
            logger.error("[CN]：[CN]2[CN]")
            return np.zeros(512 if self.dataset == "laion" else 128, dtype=np.float32)
        
        # [CN]Vector dimension
        first_result = results[server_ids[0]]['result_share']
        vector_dim = len(first_result)
        
        # [CN]
        reconstructed_vector = np.zeros(vector_dim, dtype=np.float32)
        
        # [CN]
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
            
            # [CN]
            if reconstructed > self.config.prime // 2:
                signed = reconstructed - self.config.prime
            else:
                signed = reconstructed
            
            reconstructed_vector[i] = signed / scale_factor
        
        return reconstructed_vector
    
    def _verify_result(self, node_id: int, reconstructed_vector: np.ndarray):
        """[CN]"""
        try:
            if node_id < len(self.original_nodes):
                original_vector = self.original_nodes[node_id]
                
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


def generate_markdown_report(dataset, query_details, avg_phases, avg_similarity):
    """[CN]Markdown[CN]"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    markdown = f"""# [CN]Test results[CN] - {dataset}

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

- **[CN]1 (VDPF[CN])**: {avg_phases['phase1']:.2f}[CN]
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
    parser = argparse.ArgumentParser(description='[CN]')
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
    
    client = DistributedClient(args.dataset, servers_config)
    
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
        all_similarities = []
        query_details = []
        
        # [CN]
        total_nodes = len(client.original_nodes)
        
        # [CN]
        random_nodes = random.sample(range(total_nodes), min(args.num_queries, total_nodes))
        
        logger.info(f"[CN] {len(random_nodes)} [CN]...\n")
        
        for i, node_id in enumerate(random_nodes):
            logger.info(f"[CN] {i+1}/{len(random_nodes)}: [CN] {node_id}")
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
            
            if all_similarities:
                avg_similarity = np.mean(all_similarities)
                logger.info(f"  [CN]: {avg_similarity:.6f}")
            else:
                avg_similarity = 0.0
            
            # [CN]
            if not args.no_report and query_details:
                report_file = "~/trident/distributed-deploy/distributed_result.md"
                markdown_report = generate_markdown_report(
                    args.dataset, 
                    query_details, 
                    avg_phases,
                    avg_similarity
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