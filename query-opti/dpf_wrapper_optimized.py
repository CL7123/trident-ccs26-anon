"""
优化版的DPF Wrapper，使用二进制序列化
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
    """优化版的VDPF包装器，使用二进制序列化"""
    
    def __init__(self, dataset_name: str = None):
        # 不调用父类的__init__，而是直接初始化
        self.dataset_name = dataset_name
        self.config = get_config(self.dataset_name)
        self.domain_bits = self.config.domain_bits
        self.field_size = self.config.prime
        
        # 使用标准的VDPF23
        from vdpf_23 import VDPF23
        self.vdpf = VDPF23(self.domain_bits, security_param=self.config.kappa, 
                           field_size=self.field_size, dataset_name=self.dataset_name)
        
        self.use_binary_serialization = True
        self.batch_wrapper = OptimizedBatchVDPFWrapper(self.vdpf)
        
    
    def generate_keys(self, query_type: str, position: int):
        """生成密钥并使用二进制序列化"""
        # 调用父类方法生成原始密钥
        # 使用1作为输出值（用于选择向量）
        keys = self.vdpf.gen(position, 1)
        
        if self.use_binary_serialization:
            # 使用二进制序列化，直接返回bytes
            serialized_keys = []
            for key in keys:
                serialized = BinaryKeySerializer.serialize_vdpf23_key(key)
                serialized_keys.append(serialized)  # 直接返回bytes
            return serialized_keys
        else:
            # 使用原始的pickle序列化
            return super().generate_and_serialize_keys(query_type, position)
    
    def eval_at_position(self, key, position: int, party_id: int) -> int:
        """评估VDPF在指定位置的值"""
        # 如果key是bytes，说明是二进制序列化的
        if isinstance(key, bytes):
            key = BinaryKeySerializer.deserialize_vdpf23_key(key)
        
        return self.vdpf.eval(party_id, key, position)
    
    def _serialize_key(self, key):
        """使用二进制序列化（覆盖父类方法）"""
        if self.use_binary_serialization:
            return BinaryKeySerializer.serialize_vdpf23_key(key)
        else:
            return super()._serialize_key(key)
    
    def _deserialize_key(self, serialized):
        """使用二进制反序列化（覆盖父类方法）"""
        if isinstance(serialized, bytes) and self.use_binary_serialization:
            return BinaryKeySerializer.deserialize_vdpf23_key(serialized)
        else:
            return super()._deserialize_key(serialized)
    
    def eval_batch(self, key, start_pos: int, end_pos: int, party_id: int) -> dict:
        """批量评估一个范围内的所有位置"""
        # 如果key是bytes，先反序列化
        if isinstance(key, bytes):
            key = BinaryKeySerializer.deserialize_vdpf23_key(key)
        
        return self.batch_wrapper.eval_range_batch(key, start_pos, end_pos, party_id)