
"""
VDPF包装器 - 封装(2,3)-VDPF的序列化和向量查询功能
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
    """VDPF向量查询包装器"""
    
    def __init__(self, dataset_name: str = "laion"):
        """
        初始化VDPF包装器
        
        Args:
            dataset_name: 数据集名称（使用预定义的安全配置）
        """
        self.dataset_name = dataset_name
        self.config = get_config(self.dataset_name)
        self.domain_bits = self.config.domain_bits  # 使用数据集配置的domain_bits
        self.field_size = self.config.prime
        # 创建VDPF23时传递数据集名称
        self.vdpf = VDPF23(self.domain_bits, security_param=self.config.kappa, 
                          field_size=self.field_size, dataset_name=self.dataset_name)
        
    def generate_keys(self, query_type: str, node_id: int, layer: int = None) -> List[str]:
        """
        生成VDPF密钥
        
        Args:
            query_type: 查询类型 ('node' 或 'neighbor')
            node_id: 节点ID
            layer: 层数（仅邻居查询需要）
            
        Returns:
            3个序列化的密钥字符串
        """
        # # 计算查询索引
        # if query_type == 'node':
        #     alpha = node_id
        #     # 节点查询使用较小的域（100,000个节点）
        #     if alpha >= 100000:
        #         raise ValueError(f"节点ID {alpha} 超出范围 [0, 99999]")
        # elif query_type == 'neighbor':
        #     if layer is None:
        #         raise ValueError("邻居查询需要指定layer")
        #     # 邻居查询使用线性索引
        #     alpha = node_id * 3 + layer
        #     # 检查范围（300,000个条目）
        #     if alpha >= 300000:
        #         raise ValueError(f"邻居索引 {alpha} 超出范围 [0, 299999]")
        # else:
        #     raise ValueError(f"未知的查询类型: {query_type}")
        
        alpha = node_id
        # 生成VDPF密钥
        # 使用1作为输出值（用于选择向量）
        keys = self.vdpf.gen(alpha, 1)
        
        # 序列化密钥
        serialized_keys = []
        for key in keys:
            serialized = self._serialize_key(key)
            serialized_keys.append(serialized)
        
        return serialized_keys
    
    def deserialize_and_eval(self, serialized_key: str, party_id: int) -> np.ndarray:
        """
        反序列化密钥并评估得到选择器向量
        
        Args:
            serialized_key: 序列化的密钥
            party_id: 参与方ID (1, 2, 或 3)
            
        Returns:
            选择器向量（one-hot向量的秘密份额）
        """
        # 反序列化密钥
        key = self._deserialize_key(serialized_key)
        
        # 评估VDPF
        domain_size = 2 ** self.domain_bits
        selector = np.zeros(domain_size, dtype=np.uint32)
        
        # 对整个域进行评估
        for i in range(domain_size):
            selector[i] = self.vdpf.eval(key, i, party_id)
        
        return selector
    
    def eval_at_position(self, key: VDPF23Key, position: int, party_id: int) -> int:
        """
        在特定位置评估VDPF
        
        Args:
            key: VDPF23密钥对象
            position: 评估位置
            party_id: 参与方ID (1, 2, 或 3)
            
        Returns:
            该位置的值
        """
        return self.vdpf.eval(party_id, key, position)
    
    def _serialize_key(self, key: VDPF23Key) -> str:
        """序列化VDPF密钥为base64字符串"""
        # 使用pickle序列化
        serialized_bytes = pickle.dumps(key)
        # 转换为base64以便传输
        return base64.b64encode(serialized_bytes).decode('utf-8')
    
    def _deserialize_key(self, serialized: str) -> VDPF23Key:
        """从base64字符串反序列化VDPF密钥"""
        # 从base64解码
        serialized_bytes = base64.b64decode(serialized.encode('utf-8'))
        # 使用pickle反序列化
        return pickle.loads(serialized_bytes)
    
    # def get_query_domain_size(self, query_type: str) -> int:
    #     """获取查询类型对应的域大小"""
    #     if query_type == 'node':
    #         return 100000  # 100k个节点
    #     elif query_type == 'neighbor':
    #         return 300000  # 100k * 3层
    #     else:
    #         raise ValueError(f"未知的查询类型: {query_type}")



