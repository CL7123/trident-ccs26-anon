#!/usr/bin/env python3
"""
[CN]
[CN]
"""

import sys
import os
import secrets
import time
import logging
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from collections import Counter
from enum import Enum

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from basic_functionalities import Share, MPC23SSS, DomainConfig, get_config

logger = logging.getLogger(__name__)


# ==================== [CN] ====================

@dataclass
class TripleShare:
    """Beaver[CN]"""
    triple_id: str
    server_id: int
    a_share: Share
    b_share: Share
    c_share: Share


@dataclass 
class ComputationRequest:
    """calculate[CN]"""
    computation_id: str
    x_share: Share
    y_share: Share
    triple_id: str


# ==================== Data Owner: [CN] ====================

class TripleGenerator:
    """
    Data Owner[CN]：[CN]Beaver[CN]
    [CN]：
    1. [CN]Beaver[CN]
    2. [CN]
    3. [CN]
    """
    
    def __init__(self, config: Optional[DomainConfig] = None):
        self.config = config or get_config("laion")
        self.field_size = self.config.prime
        self.mpc = MPC23SSS(self.config)
        self.generated_count = 0
        
    def generate_triples_numpy(self, count: int) -> Dict[int, np.ndarray]:
        """
        [CN]Beaver[CN]allocate[CN]（NumPy[CN]）
        
        Returns:
            Dict[server_id, np.ndarray]: [CN]servers[CN]
            [CN]: (count, 3)，[CN] [a_share, b_share, c_share]
        """
        # initialize[CN]
        server_arrays = {
            1: np.zeros((count, 3), dtype=np.uint32),
            2: np.zeros((count, 3), dtype=np.uint32),
            3: np.zeros((count, 3), dtype=np.uint32)
        }
        
        # [CN]
        print_interval = max(count // 100, 1000)  # [CN]1%[CN]1000[CN]print[CN]
        start_time = time.time()
        
        for i in range(count):
            # [CN]a, b
            a = secrets.randbelow(self.field_size)
            b = secrets.randbelow(self.field_size)
            c = (a * b) % self.field_size
            
            # [CN]
            a_shares = self.mpc.share_secret(a)
            b_shares = self.mpc.share_secret(b)
            c_shares = self.mpc.share_secret(c)
            
            # allocate[CN]
            for server_id in range(1, 4):
                server_arrays[server_id][i] = [
                    a_shares[server_id-1].value,
                    b_shares[server_id-1].value,
                    c_shares[server_id-1].value
                ]
            
            # print[CN]
            if (i + 1) % print_interval == 0 or i == count - 1:
                progress = (i + 1) / count * 100
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (count - i - 1) / rate if rate > 0 else 0
                print(f"  [CN]: {i+1:,}/{count:,} ({progress:.1f}%) - "
                      f"[CN]: {rate:.0f}/[CN] - [CN]: {eta:.0f}[CN]")
                
        logger.info(f"Generated {count} Beaver triples in NumPy format")
        return server_arrays
        
    def generate_triples(self, count: int) -> Dict[int, List[TripleShare]]:
        """
        [CN]Beaver[CN]allocate[CN]（[CN]）
        
        Returns:
            Dict[server_id, List[TripleShare]]: [CN]servers[CN]
        """
        server_shares = {1: [], 2: [], 3: []}
        
        for i in range(count):
            # [CN]a, b
            a = secrets.randbelow(self.field_size)
            b = secrets.randbelow(self.field_size)
            c = (a * b) % self.field_size
            
            # [CN]
            a_shares = self.mpc.share_secret(a)
            b_shares = self.mpc.share_secret(b)
            c_shares = self.mpc.share_secret(c)
            
            # allocate[CN]
            triple_id = f"triple_{self.generated_count}_{int(time.time()*1000)}"
            self.generated_count += 1
            
            for server_id in range(1, 4):
                triple_share = TripleShare(
                    triple_id=triple_id,
                    server_id=server_id,
                    a_share=a_shares[server_id-1],
                    b_share=b_shares[server_id-1],
                    c_share=c_shares[server_id-1]
                )
                server_shares[server_id].append(triple_share)
                
        logger.info(f"Generated {count} Beaver triples")
        return server_shares


# ==================== Server: [CN] ====================

class MultiplicationServer:
    """
    [CN]：[CN]calculate
    [CN]：
    1. [CN]
    2. calculatee[CN]f[CN]
    3. [CN]
    4. calculate[CN]
    """
    
    def __init__(self, server_id: int, config: Optional[DomainConfig] = None):
        self.server_id = server_id
        self.config = config or get_config("laion")
        self.field_size = self.config.prime
        self.triple_storage: Dict[str, TripleShare] = {}
        self.computation_cache: Dict[str, Any] = {}
        
    def load_triples(self, triples: List[TripleShare]):
        """[CN]"""
        for triple in triples:
            if triple.server_id != self.server_id:
                raise ValueError(f"Triple {triple.triple_id} not for this server")
            self.triple_storage[triple.triple_id] = triple
        logger.info(f"Server {self.server_id} loaded {len(triples)} triples")
        
    def compute_e_f_shares(self, request: ComputationRequest) -> Tuple[Share, Share]:
        """
        calculatee[CN]f[CN]
        e = x - a
        f = y - b
        """
        if request.triple_id not in self.triple_storage:
            raise ValueError(f"Triple {request.triple_id} not found")
            
        triple = self.triple_storage[request.triple_id]
        
        # calculatee_i = x_i - a_i
        e_value = (request.x_share.value - triple.a_share.value) % self.field_size
        e_share = Share(e_value, self.server_id)
        
        # calculatef_i = y_i - b_i
        f_value = (request.y_share.value - triple.b_share.value) % self.field_size
        f_share = Share(f_value, self.server_id)
        
        # [CN]calculate
        self.computation_cache[request.computation_id] = {
            'triple': triple,
            'request': request,
            'e_share': e_share,
            'f_share': f_share
        }
        
        return e_share, f_share
        
    def compute_result_share(self, computation_id: str, e: int, f: int) -> Share:
        """
        calculate[CN]
        z_i = c_i + e*b_i + f*a_i + e*f
        """
        if computation_id not in self.computation_cache:
            raise ValueError(f"Computation {computation_id} not found")
            
        cache = self.computation_cache[computation_id]
        triple = cache['triple']
        
        # z_i = c_i + e*b_i + f*a_i + e*f
        result = triple.c_share.value
        result = (result + e * triple.b_share.value) % self.field_size
        result = (result + f * triple.a_share.value) % self.field_size
        result = (result + e * f) % self.field_size
        
        # [CN]
        del self.computation_cache[computation_id]
        
        # [CN]
        del self.triple_storage[triple.triple_id]
        
        return Share(result, self.server_id)


class NumpyMultiplicationServer:
    """
    [CN]：[CN]NumPy[CN]
    """
    
    def __init__(self, server_id: int, config: Optional[DomainConfig] = None,
                 data_dir: str = "~/trident/dataset/triples"):
        self.server_id = server_id
        self.config = config or get_config("laion")
        self.field_size = self.config.prime
        self.triple_array: Optional[np.ndarray] = None
        self.used_count = 0
        self.computation_cache: Dict[str, Any] = {}
        self.data_dir = data_dir
        
        # [CN]
        self._load_local_triples()
        
    def _load_local_triples(self):
        """[CN]"""
        import glob
        
        # [CN]
        server_dir = f"{self.data_dir}/server_{self.server_id}"
        
        # [CN]
        triple_files = sorted(glob.glob(f"{server_dir}/triples_*.npy"))
        
        if not triple_files:
            logger.warning(f"Server {self.server_id}: No triple files found in {server_dir}")
            return
            
        # [CN]（[CN]）
        # [CN]: triples_10000.npy -> 10000
        latest_file = max(triple_files, 
                         key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))
        
        # [CN]
        self.triple_array = np.load(latest_file)
        self.used_count = 0
        
        logger.info(f"Server {self.server_id}: Loaded {len(self.triple_array)} triples from {latest_file}")
        
    def get_next_triple(self) -> Tuple[int, int, int]:
        """[CN]"""
        if self.triple_array is None or self.used_count >= len(self.triple_array):
            raise RuntimeError("No more triples available")
            
        triple = self.triple_array[self.used_count]
        self.used_count += 1
        return int(triple[0]), int(triple[1]), int(triple[2])
    
    def get_triple_at(self, index: int) -> Tuple[int, int, int]:
        """[CN]（[CN]）"""
        if self.triple_array is None or index >= len(self.triple_array):
            raise RuntimeError("Triple index out of range")
        triple = self.triple_array[index]
        return int(triple[0]), int(triple[1]), int(triple[2])
        
    def compute_e_f_shares(self, x_value: int, y_value: int, 
                          computation_id: str) -> Tuple[int, int]:
        """
        calculatee[CN]f[CN]（[CN]）
        """
        a, b, c = self.get_next_triple()
        
        # calculatee_i = x_i - a_i
        e_value = (x_value - a) % self.field_size
        
        # calculatef_i = y_i - b_i
        f_value = (y_value - b) % self.field_size
        
        # [CN]calculate
        self.computation_cache[computation_id] = {
            'a': a, 'b': b, 'c': c
        }
        
        return e_value, f_value
        
    def compute_result_share(self, computation_id: str, e: int, f: int) -> int:
        """
        calculate[CN]（[CN]）
        """
        if computation_id not in self.computation_cache:
            raise ValueError(f"Computation {computation_id} not found")
            
        cache = self.computation_cache[computation_id]
        a, b, c = cache['a'], cache['b'], cache['c']
        
        # z_i = c_i + e*b_i + f*a_i + e*f
        result = c
        result = (result + e * b) % self.field_size
        result = (result + f * a) % self.field_size
        result = (result + e * f) % self.field_size
        
        # [CN]
        del self.computation_cache[computation_id]
        
        return result


# ==================== Client: [CN] ====================

class MultiplicationClient:
    """
    [CN]：[CN]
    [CN]：
    1. [CN]
    2. [CN]e,f[CN]
    3. [CN]calculate
    4. [CN]
    """
    
    def __init__(self, config: Optional[DomainConfig] = None):
        self.config = config or get_config("laion")
        self.field_size = self.config.prime
        self.mpc = MPC23SSS(self.config)
        self.computation_count = 0
        
    def multiply(self, 
                 x_shares: List[Share], 
                 y_shares: List[Share],
                 servers: List[MultiplicationServer],
                 triple_id: str) -> Tuple[List[Share], Optional[int]]:
        """
        [CN]
        
        Returns:
            (result_shares, detected_malicious_server_id)
        """
        if len(x_shares) != 3 or len(y_shares) != 3:
            raise ValueError("Need exactly 3 shares")
            
        computation_id = f"comp_{self.computation_count}_{int(time.time()*1000)}"
        self.computation_count += 1
        
        # Step 1: sendcalculate[CN]
        e_f_shares = []
        for i in range(3):
            request = ComputationRequest(
                computation_id=computation_id,
                x_share=x_shares[i],
                y_share=y_shares[i],
                triple_id=triple_id
            )
            e_share, f_share = servers[i].compute_e_f_shares(request)
            e_f_shares.append((e_share, f_share))
            
        # Step 2: [CN]e[CN]f
        e_shares = [ef[0] for ef in e_f_shares]
        f_shares = [ef[1] for ef in e_f_shares]
        
        e = self._reconstruct_with_check(e_shares, "e")
        f = self._reconstruct_with_check(f_shares, "f")
        
        # Step 3: [CN]calculate[CN]
        result_shares = []
        for i in range(3):
            result_share = servers[i].compute_result_share(computation_id, e, f)
            result_shares.append(result_share)
            
        # Step 4: [CN]（[CN]）
        detected_malicious = self._verify_result(result_shares)
        
        return result_shares, detected_malicious
        
    def _reconstruct_with_check(self, shares: List[Share], label: str) -> int:
        """[CN]"""
        # [CN]
        value_12 = self.mpc.reconstruct([shares[0], shares[1]])
        value_13 = self.mpc.reconstruct([shares[0], shares[2]]) 
        value_23 = self.mpc.reconstruct([shares[1], shares[2]])
        
        # [CN]
        if value_12 == value_13 == value_23:
            return value_12
            
        # [CN]
        values = [value_12, value_13, value_23]
        counter = Counter(values)
        most_common = counter.most_common(1)[0]
        
        if most_common[1] >= 2:
            value = most_common[0]
            # [CN]
            if value_12 == value_13 == value:
                logger.warning(f"Server 3 might be malicious on {label}")
            elif value_12 == value_23 == value:
                logger.warning(f"Server 1 might be malicious on {label}")
            elif value_13 == value_23 == value:
                logger.warning(f"Server 2 might be malicious on {label}")
            return value
        else:
            raise RuntimeError(f"No consensus on {label}: {values}")
            
    def _verify_result(self, result_shares: List[Share]) -> Optional[int]:
        """[CN]"""
        # [CN]
        r12 = self.mpc.reconstruct([result_shares[0], result_shares[1]])
        r13 = self.mpc.reconstruct([result_shares[0], result_shares[2]])
        r23 = self.mpc.reconstruct([result_shares[1], result_shares[2]])
        
        # [CN]，[CN]
        if r12 == r13 == r23:
            return None
            
        # [CN]
        if r12 == r13:
            return 3  # Server 3 is malicious
        elif r12 == r23:
            return 1  # Server 1 is malicious
        elif r13 == r23:
            return 2  # Server 2 is malicious
        else:
            raise RuntimeError("Cannot identify malicious server")


# ==================== [CN] ====================

def test_secure_multiplication():
    """[CN]"""
    print("=== [CN] ===\n")
    
    config = get_config("laion")
    
    # 1. Data Owner[CN]
    print("1. Data Owner[CN]")
    generator = TripleGenerator(config)
    triple_shares = generator.generate_triples(5)  # [CN]5[CN]
    
    # 2. initialize[CN]
    print("\n2. initialize3servers")
    servers = []
    for server_id in range(1, 4):
        server = MultiplicationServer(server_id, config)
        server.load_triples(triple_shares[server_id])
        servers.append(server)
        
    # 3. [CN]
    print("\n3. [CN]: 42 × 37")
    client = MultiplicationClient(config)
    mpc = MPC23SSS(config)
    
    # [CN]
    x = 42
    y = 37
    x_shares = mpc.share_secret(x)
    y_shares = mpc.share_secret(y)
    
    # [CN]
    result_shares, malicious = client.multiply(
        x_shares, y_shares, servers, triple_shares[1][0].triple_id
    )
    
    # [CN]
    result = mpc.reconstruct(result_shares[:2])
    expected = x * y
    
    print(f"\n[CN]: {result}")
    print(f"[CN]: {expected}")
    print(f"[CN]: {'✓ [CN]' if result == expected else '✗ [CN]'}")
    print(f"[CN]: {malicious if malicious else '[CN]'}")
    
    # 4. [CN]
    print("\n\n4. [CN]（[CN]VDPF[CN]）")
    n = 10  # [CN]
    selector = [0] * n
    selector[3] = 1  # [CN]4[CN]
    data = list(range(10, 10+n))
    
    # calculate Σ(selector[i] × data[i])
    sum_result = 0
    for i in range(n):
        if i < len(triple_shares[1]):  # [CN]
            s_shares = mpc.share_secret(selector[i])
            d_shares = mpc.share_secret(data[i])
            
            r_shares, _ = client.multiply(
                s_shares, d_shares, servers, triple_shares[1][i].triple_id
            )
            
            r = mpc.reconstruct(r_shares[:2])
            sum_result = (sum_result + r) % config.prime
            
    print(f"[CN]: {sum_result}")
    print(f"[CN]: {data[3]}")
    print(f"[CN]: {'✓ [CN]' if sum_result == data[3] else '✗ [CN]'}")
    
    print("\n=== [CN] ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_secure_multiplication()