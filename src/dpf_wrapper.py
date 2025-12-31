
"""
VDPF[CN] - [CN](2,3)-VDPF[CN]
"""

import sys
import os
import pickle
import base64
from typing import Tuple, List, Dict, Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('~/trident/standardDPF')

from vdpf_23 import VDPF23, VDPF23Key
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('~/trident/src')
from domain_config import get_config


class VDPFVectorWrapper:
    """VDPF[CN]"""
    
    def __init__(self, dataset_name: str = "laion"):
        """
        initializeVDPF[CN]
        
        Args:
            dataset_name: Dataset[CN]（[CN]）
        """
        self.dataset_name = dataset_name
        self.config = get_config(self.dataset_name)
        self.domain_bits = self.config.domain_bits  # [CN]Dataset[CN]domain_bits
        self.field_size = self.config.prime
        # createVDPF23[CN]Dataset[CN]
        self.vdpf = VDPF23(self.domain_bits, security_param=self.config.kappa, 
                          field_size=self.field_size, dataset_name=self.dataset_name)
        
    def generate_keys(self, query_type: str, node_id: int, layer: int = None) -> List[str]:
        """
        [CN]VDPF[CN]
        
        Args:
            query_type: [CN] ('node' [CN] 'neighbor')
            node_id: [CN]ID
            layer: [CN]（[CN]）
            
        Returns:
            3[CN]
        """
        # # calculate[CN]
        # if query_type == 'node':
        #     alpha = node_id
        #     # [CN]（100,000[CN]）
        #     if alpha >= 100000:
        #         raise ValueError(f"[CN]ID {alpha} [CN] [0, 99999]")
        # elif query_type == 'neighbor':
        #     if layer is None:
        #         raise ValueError("[CN]layer")
        #     # [CN]
        #     alpha = node_id * 3 + layer
        #     # [CN]（300,000[CN]）
        #     if alpha >= 300000:
        #         raise ValueError(f"[CN] {alpha} [CN] [0, 299999]")
        # else:
        #     raise ValueError(f"[CN]: {query_type}")
        
        alpha = node_id
        # [CN]VDPF[CN]
        # [CN]1[CN]（[CN]）
        keys = self.vdpf.gen(alpha, 1)
        
        # [CN]
        serialized_keys = []
        for key in keys:
            serialized = self._serialize_key(key)
            serialized_keys.append(serialized)
        
        return serialized_keys
    
    def deserialize_and_eval(self, serialized_key: str, party_id: int) -> np.ndarray:
        """
        Deserialize keys[CN]
        
        Args:
            serialized_key: [CN]
            party_id: [CN]ID (1, 2, [CN] 3)
            
        Returns:
            [CN]（one-hot[CN]）
        """
        # Deserialize keys
        key = self._deserialize_key(serialized_key)
        
        # [CN]VDPF
        domain_size = 2 ** self.domain_bits
        selector = np.zeros(domain_size, dtype=np.uint32)
        
        # [CN]
        for i in range(domain_size):
            selector[i] = self.vdpf.eval(key, i, party_id)
        
        return selector
    
    def eval_at_position(self, key: VDPF23Key, position: int, party_id: int) -> int:
        """
        [CN]VDPF
        
        Args:
            key: VDPF23[CN]
            position: [CN]
            party_id: [CN]ID (1, 2, [CN] 3)
            
        Returns:
            [CN]
        """
        return self.vdpf.eval(party_id, key, position)
    
    def _serialize_key(self, key: VDPF23Key) -> str:
        """[CN]VDPF[CN]base64[CN]"""
        # [CN]pickle[CN]
        serialized_bytes = pickle.dumps(key)
        # [CN]base64[CN]
        return base64.b64encode(serialized_bytes).decode('utf-8')
    
    def _deserialize_key(self, serialized: str) -> VDPF23Key:
        """[CN]base64[CN]VDPF[CN]"""
        # [CN]base64[CN]
        serialized_bytes = base64.b64decode(serialized.encode('utf-8'))
        # [CN]pickle[CN]
        return pickle.loads(serialized_bytes)
    
    # def get_query_domain_size(self, query_type: str) -> int:
    #     """[CN]"""
    #     if query_type == 'node':
    #         return 100000  # 100k[CN]
    #     elif query_type == 'neighbor':
    #         return 300000  # 100k * 3[CN]
    #     else:
    #         raise ValueError(f"[CN]: {query_type}")



