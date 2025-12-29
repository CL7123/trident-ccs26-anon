"""
VDPF批量评估优化器
通过批量处理减少函数调用开销和提高缓存效率
"""
import numpy as np
from typing import List, Dict
import sys
sys.path.append('~/trident/standardDPF')

from vdpf_23 import VDPF23, VDPF23Key


class BatchVDPF23:
    """批量优化版的VDPF23"""
    
    def __init__(self, vdpf23_instance: VDPF23):
        self.vdpf = vdpf23_instance
        
    def eval_batch(self, b: int, fb: VDPF23Key, positions: List[int]) -> Dict[int, int]:
        """
        批量评估VDPF在多个位置的值
        
        Args:
            b: Party ID (1, 2, or 3)
            fb: The key for party b
            positions: List of positions to evaluate
            
        Returns:
            Dictionary mapping position to value
        """
        # 预先解析密钥（只做一次）
        g = fb.g_key
        k = fb.k_key
        
        # 预先计算party相关的常量（只做一次）
        if b == 1:
            b_g, b_k = 0, 0
        elif b == 2:
            b_g, b_k = 1, 0
        else:  # b == 3
            b_g, b_k = 1, 1
        
        results = {}
        
        # 批量处理，提高缓存局部性
        for x in positions:
            # 评估VDPF+
            y_g = self.vdpf.vdpf_plus_g.eval(b_g, g, x)
            y_k = self.vdpf.vdpf_plus_k.eval(b_k, k, x)
            
            # 计算最终结果
            y_xor = y_g ^ y_k
            y = self.vdpf.op_embed_mult(b, y_xor)
            
            results[x] = y
            
        return results
    
    def eval_batch_vectorized(self, b: int, fb: VDPF23Key, positions: np.ndarray) -> np.ndarray:
        """
        向量化批量评估（如果底层支持）
        
        Args:
            b: Party ID (1, 2, or 3)
            fb: The key for party b
            positions: Numpy array of positions
            
        Returns:
            Numpy array of values
        """
        # 预先解析密钥
        g = fb.g_key
        k = fb.k_key
        
        # 预先计算party相关的常量
        if b == 1:
            b_g, b_k = 0, 0
        elif b == 2:
            b_g, b_k = 1, 0
        else:  # b == 3
            b_g, b_k = 1, 1
        
        # 预分配结果数组
        results = np.zeros(len(positions), dtype=np.uint64)
        
        # 批量评估
        for i, x in enumerate(positions):
            y_g = self.vdpf.vdpf_plus_g.eval(b_g, g, x)
            y_k = self.vdpf.vdpf_plus_k.eval(b_k, k, x)
            y_xor = y_g ^ y_k
            results[i] = self.vdpf.op_embed_mult(b, y_xor)
            
        return results


class OptimizedBatchVDPFWrapper:
    """优化的批量VDPF包装器"""
    
    def __init__(self, vdpf_instance):
        self.vdpf = vdpf_instance
        self.batch_evaluator = BatchVDPF23(vdpf_instance)
        
    def eval_positions_batch(self, key, positions: List[int], party_id: int) -> Dict[int, int]:
        """
        批量评估多个位置
        
        Args:
            key: VDPF密钥
            positions: 位置列表
            party_id: Party ID
            
        Returns:
            位置到值的映射
        """
        return self.batch_evaluator.eval_batch(party_id, key, positions)
    
    def eval_range_batch(self, key, start: int, end: int, party_id: int) -> Dict[int, int]:
        """
        批量评估一个范围内的所有位置
        
        Args:
            key: VDPF密钥
            start: 起始位置（包含）
            end: 结束位置（不包含）
            party_id: Party ID
            
        Returns:
            位置到值的映射
        """
        positions = list(range(start, end))
        return self.batch_evaluator.eval_batch(party_id, key, positions)