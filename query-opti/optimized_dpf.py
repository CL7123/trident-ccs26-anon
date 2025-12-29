#!/usr/bin/env python3
"""
优化的DPF实现，使用PRG缓存减少重复计算
"""
import hashlib
from typing import Tuple, Dict
import sys
sys.path.append('~/trident/standardDPF')

from standard_dpf import StandardDPF, DPFKey


class CachedPRG:
    """带缓存的PRG实现"""
    
    def __init__(self, max_cache_size=10000):
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.hits = 0
        self.misses = 0
        
    def expand(self, seed: bytes, seed_len: int) -> Tuple[bytes, bytes, int, int]:
        """带缓存的PRG扩展"""
        # 使用seed作为缓存键
        cache_key = seed
        
        if cache_key in self.cache:
            self.hits += 1
            return self.cache[cache_key]
        
        self.misses += 1
        
        # 计算PRG扩展
        # 使用单次SHA-512代替两次SHA-256
        h = hashlib.sha512()
        h.update(seed)
        output = h.digest()
        
        # 分割输出
        s_left = output[:seed_len]
        s_right = output[32:32+seed_len]
        t_left = output[seed_len] & 1
        t_right = output[32+seed_len] & 1
        
        # 缓存结果
        if len(self.cache) < self.max_cache_size:
            self.cache[cache_key] = (s_left, s_right, t_left, t_right)
        
        return s_left, s_right, t_left, t_right
    
    def get_stats(self):
        """获取缓存统计"""
        total = self.hits + self.misses
        hit_rate = self.hits / total * 100 if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }


class OptimizedDPF(StandardDPF):
    """优化的DPF实现"""
    
    def __init__(self, domain_bits: int):
        super().__init__(domain_bits)
        self.prg_cache = CachedPRG()
        
    def _dpf_prg(self, seed: bytes) -> Tuple[bytes, bytes, int, int]:
        """使用缓存的PRG"""
        return self.prg_cache.expand(seed, self.seed_len)
    
    def eval_batch(self, key: DPFKey, positions: list) -> dict:
        """批量评估多个位置"""
        results = {}
        
        # 按照共同前缀分组，提高缓存命中率
        # 先排序位置，相邻的位置更可能共享路径前缀
        sorted_positions = sorted(positions)
        
        for x in sorted_positions:
            results[x] = self.eval(key, x)
        
        return results
    
    def get_prg_stats(self):
        """获取PRG缓存统计"""
        return self.prg_cache.get_stats()


class OptimizedVDPF:
    """优化的VDPF实现，使用优化的DPF"""
    
    def __init__(self, domain_bits: int):
        self.domain_bits = domain_bits
        # 使用优化的DPF
        self.dpf = OptimizedDPF(domain_bits)
        
    def eval_batch_with_stats(self, key, positions: list) -> Tuple[dict, dict]:
        """批量评估并返回统计信息"""
        import time
        
        start_time = time.time()
        
        # 重置统计
        self.dpf.prg_cache.hits = 0
        self.dpf.prg_cache.misses = 0
        
        # 批量评估
        results = self.dpf.eval_batch(key, positions)
        
        eval_time = time.time() - start_time
        stats = self.dpf.get_prg_stats()
        stats['total_time'] = eval_time
        stats['positions_evaluated'] = len(positions)
        stats['avg_time_per_position'] = eval_time / len(positions) * 1000  # ms
        
        return results, stats