
"""
VDPF wrapper - Encapsulates (2,3)-VDPF serialization and vector query functionality
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
    """VDPF vector query wrapper"""

    def __init__(self, dataset_name: str = "laion"):
        """
        Initialize VDPF wrapper

        Args:
            dataset_name: Dataset name (uses predefined security configuration)
        """
        self.dataset_name = dataset_name
        self.config = get_config(self.dataset_name)
        self.domain_bits = self.config.domain_bits  # Use domain_bits from dataset config
        self.field_size = self.config.prime
        # Pass dataset name when creating VDPF23
        self.vdpf = VDPF23(self.domain_bits, security_param=self.config.kappa,
                          field_size=self.field_size, dataset_name=self.dataset_name)

    def generate_keys(self, query_type: str, node_id: int, layer: int = None) -> List[str]:
        """
        Generate VDPF keys

        Args:
            query_type: Query type ('node' or 'neighbor')
            node_id: Node ID
            layer: Layer number (only needed for neighbor queries)

        Returns:
            3 serialized key strings
        """
        # # Calculate query index
        # if query_type == 'node':
        #     alpha = node_id
        #     # Node query uses smaller domain (100,000 nodes)
        #     if alpha >= 100000:
        #         raise ValueError(f"Node ID {alpha} out of range [0, 99999]")
        # elif query_type == 'neighbor':
        #     if layer is None:
        #         raise ValueError("Neighbor query requires layer specification")
        #     # Neighbor query uses linear index
        #     alpha = node_id * 3 + layer
        #     # Check range (300,000 entries)
        #     if alpha >= 300000:
        #         raise ValueError(f"Neighbor index {alpha} out of range [0, 299999]")
        # else:
        #     raise ValueError(f"Unknown query type: {query_type}")

        alpha = node_id
        # Generate VDPF keys
        # Use 1 as output value (for vector selection)
        keys = self.vdpf.gen(alpha, 1)

        # Serialize keys
        serialized_keys = []
        for key in keys:
            serialized = self._serialize_key(key)
            serialized_keys.append(serialized)

        return serialized_keys

    def deserialize_and_eval(self, serialized_key: str, party_id: int) -> np.ndarray:
        """
        Deserialize key and evaluate to get selector vector

        Args:
            serialized_key: Serialized key
            party_id: Party ID (1, 2, or 3)

        Returns:
            Selector vector (secret share of one-hot vector)
        """
        # Deserialize key
        key = self._deserialize_key(serialized_key)

        # Evaluate VDPF
        domain_size = 2 ** self.domain_bits
        selector = np.zeros(domain_size, dtype=np.uint32)

        # Evaluate over entire domain
        for i in range(domain_size):
            selector[i] = self.vdpf.eval(key, i, party_id)

        return selector

    def eval_at_position(self, key: VDPF23Key, position: int, party_id: int) -> int:
        """
        Evaluate VDPF at specific position

        Args:
            key: VDPF23 key object
            position: Evaluation position
            party_id: Party ID (1, 2, or 3)

        Returns:
            Value at this position
        """
        return self.vdpf.eval(party_id, key, position)

    def _serialize_key(self, key: VDPF23Key) -> str:
        """Serialize VDPF key to base64 string"""
        # Use pickle for serialization
        serialized_bytes = pickle.dumps(key)
        # Convert to base64 for transmission
        return base64.b64encode(serialized_bytes).decode('utf-8')

    def _deserialize_key(self, serialized: str) -> VDPF23Key:
        """Deserialize VDPF key from base64 string"""
        # Decode from base64
        serialized_bytes = base64.b64decode(serialized.encode('utf-8'))
        # Use pickle to deserialize
        return pickle.loads(serialized_bytes)

    # def get_query_domain_size(self, query_type: str) -> int:
    #     """Get domain size for query type"""
    #     if query_type == 'node':
    #         return 100000  # 100k nodes
    #     elif query_type == 'neighbor':
    #         return 300000  # 100k * 3 layers
    #     else:
    #         raise ValueError(f"Unknown query type: {query_type}")



