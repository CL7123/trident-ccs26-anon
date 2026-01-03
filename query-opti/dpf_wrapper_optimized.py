"""
Optimized DPF Wrapper using binary serialization
"""
import sys
sys.path.append('~/trident/standardDPF')
sys.path.append('~/trident/src')
sys.path.append('~/trident/query-opti')

from dpf_wrapper import VDPFVectorWrapper
from binary_serializer import BinaryKeySerializer
from vdpf_batch_optimizer import OptimizedBatchVDPFWrapper
from domain_config import get_config


class OptimizedVDPFVectorWrapper(VDPFVectorWrapper):
    """Optimized VDPF wrapper using binary serialization"""

    def __init__(self, dataset_name: str = None):
        # Don't call parent class __init__, initialize directly instead
        self.dataset_name = dataset_name
        self.config = get_config(self.dataset_name)
        self.domain_bits = self.config.domain_bits
        self.field_size = self.config.prime

        # Use standard VDPF23
        from vdpf_23 import VDPF23
        self.vdpf = VDPF23(self.domain_bits, security_param=self.config.kappa,
                           field_size=self.field_size, dataset_name=self.dataset_name)

        self.use_binary_serialization = True
        self.batch_wrapper = OptimizedBatchVDPFWrapper(self.vdpf)
        


    def generate_keys(self, query_type: str, position: int):
        """Generate keys using binary serialization"""
        # Call parent class method to generate original keys
        # Use 1 as output value (for selecting vectors)
        keys = self.vdpf.gen(position, 1)

        if self.use_binary_serialization:
            # Use binary serialization, return bytes directly
            serialized_keys = []
            for key in keys:
                serialized = BinaryKeySerializer.serialize_vdpf23_key(key)
                serialized_keys.append(serialized)  # Return bytes directly
            return serialized_keys
        else:
            # Use original pickle serialization
            return super().generate_and_serialize_keys(query_type, position)


    def eval_at_position(self, key, position: int, party_id: int) -> int:
        """Evaluate VDPF value at specified position"""
        # If key is bytes, it means it's binary serialized
        if isinstance(key, bytes):
            key = BinaryKeySerializer.deserialize_vdpf23_key(key)

        return self.vdpf.eval(party_id, key, position)


    def _serialize_key(self, key):
        """Use binary serialization (override parent class method)"""
        if self.use_binary_serialization:
            return BinaryKeySerializer.serialize_vdpf23_key(key)
        else:
            return super()._serialize_key(key)

    def _deserialize_key(self, serialized):
        """Use binary deserialization (override parent class method)"""
        if isinstance(serialized, bytes) and self.use_binary_serialization:
            return BinaryKeySerializer.deserialize_vdpf23_key(serialized)
        else:
            return super()._deserialize_key(serialized)

    def eval_batch(self, key, start_pos: int, end_pos: int, party_id: int) -> dict:
        """Batch evaluate all positions in a range"""
        # If key is bytes, deserialize first
        if isinstance(key, bytes):
            key = BinaryKeySerializer.deserialize_vdpf23_key(key)

        return self.batch_wrapper.eval_range_batch(key, start_pos, end_pos, party_id)